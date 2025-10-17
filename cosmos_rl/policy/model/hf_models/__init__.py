# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
import inspect
from torch import nn
from safetensors import safe_open
from transformers.utils import quantization_config as transformers_quantization_config
from functools import partial, cached_property
from typing import Tuple, List, Optional, Callable

from transformers import AutoConfig, AutoModel
from cosmos_rl.utils.util import (
    clear_weight_name,
    safe_deep_getattr,
    load_model_class_by_config,
    reverse_hf_checkpoint_mapping,
    resolve_model_path,
)
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.hf_models.weight_converter import convert_weight_from_hf
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.hf_models.patch import (
    pre_hf_models_patch,
    post_hf_models_patch,
)


@ModelRegistry.register(HFModelWeightMapper)
class HFModel(BaseModel):
    """
    HFModel Module

    Args:
        hf_config : Model configuration arguments.
        model: model loaded from hf.

    Attributes:
        hf_config : Model configuration arguments.
        model: model loaded from hf.
    """

    @staticmethod
    def supported_model_types():
        return [COSMOS_HF_MODEL_TYPES]

    def __init__(
        self, hf_config, model, model_class, is_vlm=False, need_dequantization=False
    ):
        super().__init__(hf_config)
        self.hf_config = hf_config
        self.model = model
        self.model = self.model.to(dtype=hf_config.torch_dtype)
        self.model_class = model_class
        self.is_vlm = is_vlm
        self.need_dequantization = need_dequantization
        if getattr(model, "_checkpoint_conversion_mapping", None):
            if hf_config.model_type in ["R"]:
                logger.warning(
                    f"{hf_config.model_type}'s checkpoint_conversion_mapping do not take effect, "
                    "skip reverse_hf_conversion_mapping"
                )
            else:
                # Reverse HuggingFace checkpoint conversion mapping to align with VLLM weight naming convention
                self.weight_mapper.reverse_hf_conversion_mapping = (
                    reverse_hf_checkpoint_mapping(model._checkpoint_conversion_mapping)
                )
                logger.info(
                    f"reverse_hf_conversion_mapping={self.weight_mapper.reverse_hf_conversion_mapping}"
                )

    @cached_property
    def model_forward_valid_kwargs(self):
        sig = inspect.signature(self.model.forward)
        return sig.parameters.keys()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        kwargs_filtered = {
            k: v for k, v in kwargs.items() if k in self.model_forward_valid_kwargs
        }

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            *args,
            **kwargs_filtered,
        )
        return out.logits

    @property
    def image_token_id(self):
        # Retrieve image token ID from either image_token_id or image_token_index attribute
        image_token_id = None
        if self.is_vlm:
            image_token_id = getattr(self.hf_config, "image_token_id", None) or getattr(
                self.vision_config, "image_token_id", None
            )
            if image_token_id is None:
                image_token_id = getattr(
                    self.hf_config, "image_token_index", None
                ) or getattr(self.vision_config, "image_token_index", None)
            if image_token_id is None:
                raise ValueError(f"Can not get image token id from {self.hf_config}")
        return image_token_id

    @property
    def video_token_id(self):
        video_token_id = None
        if self.is_vlm:
            video_token_id = getattr(self.hf_config, "video_token_id", None) or getattr(
                self.vision_config, "video_token_id", None
            )
            if video_token_id is None:
                video_token_id = getattr(
                    self.hf_config, "video_token_index", None
                ) or getattr(self.vision_config, "video_token_index", None)
        return video_token_id

    @property
    def lm_layers(self):
        lm_layers = None
        sub_lm_model = getattr(self.language_model, "model", None)
        if sub_lm_model is None:
            lm_layers = self.language_model.layers
        else:
            lm_layers = sub_lm_model.layers
        assert (
            lm_layers is not None
        ), f"Can not get lm layers from {self.language_model}"
        return lm_layers

    @property
    def vision_layers(self):
        vision_layers = None
        if self.vision_model is not None:
            for path in [
                "blocks",  # ClipVisionModel(Llava)
                "vision_model.encoder.layers",  # SiglipVisionModel(Gemma)
                "transformer.layers",  # PixtralVisionModel（Mistral）
                "model.layers",  # Llama4VisionModel
                "encoder.layer",  # InternVLVisionModel(qwen)
                "encoder.layers",  # InternVLVisionModel(gpt-oss)
            ]:
                vision_layers = safe_deep_getattr(self.vision_model, path, None)
                if vision_layers is not None:
                    break
            assert (
                vision_layers is not None
            ), f"Can not get vision layers from {self.vision_model}."
        return vision_layers

    @property
    def n_lm_layers(self):
        n_lm_layers = 0
        if hasattr(self.text_config, "num_hidden_layers"):
            n_lm_layers = self.text_config.num_hidden_layers
        else:
            logger.warning(f"Can not get num of llm layers from {self.text_config}.")
        return n_lm_layers

    @property
    def n_vision_layers(self):
        n_vision_layers = 0
        if hasattr(self.vision_config, "num_hidden_layers"):
            n_vision_layers = self.vision_config.num_hidden_layers
        # qwen2.5-vl
        elif hasattr(self.vision_config, "depth"):
            n_vision_layers = self.vision_config.depth
        else:
            logger.warning(
                f"Can not get num of vision model layers from {self.vision_config}."
            )
        return n_vision_layers

    @property
    def text_config(self):
        text_config = None
        if self.is_vlm:
            if hasattr(self.hf_config, "text_config"):
                text_config = self.hf_config.text_config
            elif hasattr(self.hf_config, "llm_config"):
                text_config = self.hf_config.llm_config
            else:
                logger.warning(f"Can not get text config from {self.hf_config}.")
                text_config = self.hf_config
        else:
            text_config = self.hf_config
        return text_config

    @property
    def vision_config(self):
        return self.hf_config.vision_config if self.is_vlm else None

    @property
    def language_model(self):
        language_model = None
        if self.is_vlm:
            if hasattr(self.model, "language_model"):
                language_model = self.model.language_model
            elif hasattr(self.model, "model"):
                language_model = self.model.model
            else:
                logger.warning(f"Can not get language model from {self.model}.")
        else:
            language_model = self.model
        return language_model

    @property
    def embed_tokens(self):
        embed_tokens = getattr(self.language_model, "embed_tokens", None)
        if embed_tokens is None:
            embed_tokens = safe_deep_getattr(
                self.language_model, "model.embed_tokens", None
            )
        if embed_tokens is None:
            raise ValueError(f"Can not get embed tokens from {self.language_model}")
        return embed_tokens

    @property
    def vision_model(self):
        vision_model = None
        if self.is_vlm:
            # Extract vision model from various possible attribute names
            if hasattr(self.model, "vision_tower"):
                vision_model = self.model.vision_tower
            elif hasattr(self.model, "visual"):
                vision_model = self.model.visual
            elif hasattr(self.model, "vision_model"):
                vision_model = self.model.vision_model
            else:
                raise ValueError(f"Can not get vision model from {self.model}")
        return vision_model

    @property
    def multi_modal_projector(self):
        multi_modal_projector = None
        if self.is_vlm:
            if hasattr(self.model, "multi_modal_projector"):
                multi_modal_projector = self.model.multi_modal_projector
            elif hasattr(self.model, "mlp1"):
                # Handle InternVL architecture's multi-modal projector naming
                multi_modal_projector = self.model.mlp1

        return multi_modal_projector

    @property
    def delay_cp_slice_inputs(self):
        return self.is_vlm

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        # Named buffers will be reset during the load_hf_weights process
        return

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.hf_models.parallelize import parallelize

        return parallelize, self

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert False, "Pipeline split is not supported for HFModel"

    def reset_named_buffers(self, hf_model=None, model_name_or_path=None):
        reset_success = False
        if hf_model is not None:
            # copy named buffers from hf_model to self.model
            hf_named_buffers = {k: v for k, v in hf_model.named_buffers()}
            for name, cosmos_hf_buffer in self.model.named_buffers():
                assert name in hf_named_buffers, f"Buffer {name} not found in hf model"
                hf_buf = hf_named_buffers[name].to(
                    device=cosmos_hf_buffer.device, dtype=cosmos_hf_buffer.dtype
                )
                cosmos_hf_buffer.data.copy_(hf_buf.data)
            reset_success = True
        else:
            assert (
                model_name_or_path is not None
            ), "model_name_or_path is required for resetting named buffers"
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )

            # Load only first 2 layers instead of full model to extract named buffers efficiently.
            # Most buffers (e.g., inv_freq) are initialized in the model constructor, making this approach sufficient.
            num_lm_layers_to_load = 2
            if self.is_vlm:
                if hasattr(config, "text_config") and hasattr(
                    config.text_config, "num_hidden_layers"
                ):
                    config.text_config.num_hidden_layers = num_lm_layers_to_load
                    config.text_config.max_position_embeddings = (
                        self.hf_config.max_position_embeddings
                    )
                elif hasattr(config, "llm_config") and hasattr(
                    config.llm_config, "num_hidden_layers"
                ):
                    config.llm_config.num_hidden_layers = num_lm_layers_to_load
                    config.llm_config.max_position_embeddings = (
                        self.hf_config.max_position_embeddings
                    )
                else:
                    raise ValueError(f"Can not get text config from {config}")
            else:
                if hasattr(config, "num_hidden_layers"):
                    config.num_hidden_layers = num_lm_layers_to_load
                    config.max_position_embeddings = (
                        self.hf_config.max_position_embeddings
                    )
                else:
                    raise ValueError(f"Can not get num of llm layers from {config}")
            # Attempt to load partial model to extract all named buffers
            try:
                if isinstance(self.model_class, AutoModel):
                    hf_model = AutoModel.from_config(config)
                else:
                    hf_model = self.model_class._from_config(config)
                hf_named_buffers = [name for name, _ in hf_model.named_buffers()]
                self_named_buffers = [name for name, _ in self.model.named_buffers()]
                num_equal = len(hf_named_buffers) == len(self_named_buffers)
                if not num_equal:
                    # Check if the buffers are registered in the layers
                    is_buffer_registered_in_layers = any(
                        "layers." in name for name in hf_named_buffers
                    )
                    if (self.n_lm_layers - num_lm_layers_to_load) == (
                        len(self_named_buffers) - len(hf_named_buffers)
                    ) and is_buffer_registered_in_layers:
                        hf_buffer_in_layers = [
                            buffer
                            for name, buffer in hf_model.named_buffers()
                            if "layers." in name
                        ]
                        first_buffer = hf_buffer_in_layers[0]
                        all_same = True
                        # Verify that all layer buffers contain identical values
                        for buffer in hf_buffer_in_layers[1:]:
                            if not torch.equal(
                                buffer,
                                first_buffer.to(
                                    device=buffer.device, dtype=buffer.dtype
                                ),
                            ):
                                all_same = False
                                break
                        # If all buffers in the layers are the same, we can repeat the first layer's
                        # buffer to the rest of the layers
                        if all_same:
                            cosmos_buffer_in_layers = [
                                buffer
                                for name, buffer in self.model.named_buffers()
                                if "layers." in name
                            ]
                            hf_first_layer_buffer = first_buffer.to(
                                device=cosmos_buffer_in_layers[0].device,
                                dtype=cosmos_buffer_in_layers[0].dtype,
                            )
                            for buffer in cosmos_buffer_in_layers:
                                buffer.data.copy_(hf_first_layer_buffer.data)

                            hf_named_buffers_not_in_layers = {
                                k: v
                                for k, v in hf_model.named_buffers()
                                if "layers." not in k
                            }
                            for name, cosmos_hf_buffer in self.model.named_buffers():
                                if "layers." in name:
                                    continue
                                assert (
                                    name in hf_named_buffers_not_in_layers
                                ), f"Buffer {name} not found in hf model"
                                hf_buf = hf_named_buffers_not_in_layers[name].to(
                                    device=cosmos_hf_buffer.device,
                                    dtype=cosmos_hf_buffer.dtype,
                                )
                                cosmos_hf_buffer.data.copy_(hf_buf.data)

                            return True
                        else:
                            logger.warning(
                                f"Failed to reset named buffers from {model_name_or_path}: buffer names mismatch {self_named_buffers} != {hf_named_buffers}"
                            )
                    else:
                        logger.warning(
                            f"Failed to reset named buffers from {model_name_or_path}: num of buffers mismatch {len(self_named_buffers)} != {len(hf_named_buffers)}"
                        )
                        return False

                reset_success = self.reset_named_buffers(hf_model=hf_model)
            except Exception as e:
                logger.error(
                    f"Failed to reset named buffers from {model_name_or_path}: {e}"
                )
                reset_success = False

        return reset_success

    def load_hf_weights_from_safetensors(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        model_type = self.hf_config.model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        self_state_dict = self.model.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}
        lm_head_weight_key = None
        embed_tokens_weight_key = None
        # Find the lm_head and embed_tokens weight keys in the state dict
        for k in self_state_dict.keys():
            if "embed_tokens" in k:
                embed_tokens_weight_key = k
                if lm_head_weight_key is not None:
                    break
            if "lm_head" in k:
                lm_head_weight_key = k
                if embed_tokens_weight_key is not None:
                    break
        assert (
            lm_head_weight_key is not None and embed_tokens_weight_key is not None
        ), "lm_head and embed_tokens weight keys not found in the state dict"
        weights_of_ckpt_names = set()
        reserved = {}
        hf_checkpoint_conversion_mapping = getattr(
            self.model, "_checkpoint_conversion_mapping", None
        )
        for f in safetensors_files:
            weights_of_ckpt = {}
            ckpt = safe_open(
                os.path.join(model_path, f), framework="pt", device=str(device)
            )
            keys = ckpt.keys()
            for name in keys:
                ckpt_tensor = ckpt.get_tensor(name)
                if hf_checkpoint_conversion_mapping is not None:
                    for (
                        pattern,
                        replacement,
                    ) in hf_checkpoint_conversion_mapping.items():
                        if re.match(pattern, name):
                            name = re.sub(pattern, replacement, name)
                            break
                weights_of_ckpt[name] = ckpt_tensor
                weights_of_ckpt_names.add(name)
                if name == embed_tokens_weight_key:
                    reserved[name] = ckpt_tensor

            for name in weights_of_ckpt.keys():
                tensor = weights_of_ckpt[name]
                dest_name, shared_weight = convert_weight_from_hf(
                    tensor, name, model_type, parallel_dims
                )

                target_tensor = self_state_dict[dest_name]
                is_dist_tensor = isinstance(
                    target_tensor, torch.distributed.tensor.DTensor
                )
                local_view = (
                    target_tensor.to_local() if is_dist_tensor else target_tensor
                )
                assert (
                    local_view.shape == shared_weight.shape
                ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name}"
                with torch.no_grad():
                    local_view.data.copy_(shared_weight)

        if (
            lm_head_weight_key not in weights_of_ckpt_names
            and embed_tokens_weight_key in weights_of_ckpt_names
        ):
            # Handle weight tying: lm_head shares weights with embed_tokens
            name = lm_head_weight_key
            assert embed_tokens_weight_key in reserved
            tensor = reserved[embed_tokens_weight_key]
            dest_name, shared_weight = convert_weight_from_hf(
                tensor, name, model_type, parallel_dims
            )
            if dest_name in self_state_dict:
                target_tensor = self_state_dict[dest_name]
                is_dist_tensor = isinstance(
                    target_tensor, torch.distributed.tensor.DTensor
                )
                local_view = (
                    target_tensor.to_local() if is_dist_tensor else target_tensor
                )
                assert (
                    local_view.shape == shared_weight.shape
                ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name}"
                with torch.no_grad():
                    local_view.data.copy_(shared_weight.to(device))

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_path (str): Path to the HuggingFace model.
            parallel_dims (ParallelDims): Parallel dimensions definition.
        """
        model_type = self.hf_config.model_type
        dtype = self.hf_config.torch_dtype
        kwargs = {
            "config": self.hf_config,
            "revision": revision,
            "trust_remote_code": True,
        }
        if self.need_dequantization:
            quantization_config = self.hf_config.quantization_config
            mxfp4_quantization_config = transformers_quantization_config.Mxfp4Config(
                dequantize=True,
                modules_to_not_convert=quantization_config["modules_to_not_convert"],
            )
            kwargs["quantization_config"] = mxfp4_quantization_config

        # Configure gradient checkpointing if enabled
        if self._gradient_checkpointing_enabled:
            self.model.gradient_checkpointing_enable()
            assert (
                self.model.is_gradient_checkpointing
            ), "Gradient checkpointing is not enabled"
            logger.info("Enabled gradient checkpointing for HFModel")

        # Use from_pretrained loading in two scenarios:
        # 1. Model requires dequantization (e.g., gpt-oss)
        # 2. Named buffer reinitialization failed
        load_hf_weights_from_pretrained = (
            self.need_dequantization
            or not self.reset_named_buffers(model_name_or_path=model_name_or_path)
        )

        if not load_hf_weights_from_pretrained:
            return self.load_hf_weights_from_safetensors(
                model_name_or_path,
                parallel_dims,
                device,
                revision,
            )

        logger.warning(
            "Loading weights via from_pretrained method - this may take considerable time."
        )

        hf_model = self.model_class.from_pretrained(
            model_name_or_path,
            **kwargs,
        ).to(device="cpu", dtype=dtype)

        self.reset_named_buffers(hf_model=hf_model)

        hf_state_dict = hf_model.state_dict()

        self_state_dict = self.model.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}

        for name, tensor in hf_state_dict.items():
            dest_name, shared_weight = convert_weight_from_hf(
                tensor, name, model_type, parallel_dims
            )

            target_tensor = self_state_dict[dest_name]
            is_dist_tensor = isinstance(target_tensor, torch.distributed.tensor.DTensor)
            local_view = target_tensor.to_local() if is_dist_tensor else target_tensor
            assert (
                local_view.shape == shared_weight.shape
            ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name}"
            with torch.no_grad():
                local_view.data.copy_(shared_weight.to(device))

        del hf_model

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        position_ids = None
        inputs = kwargs["input_ids"]
        seq_dim_idx = 1 if not self.is_vlm else 2
        return position_ids, inputs, seq_dim_idx

    def separate_model_parts(self) -> List[nn.Module]:
        if self.is_vlm:
            model_parts = [self.language_model, self.vision_model]
            if self.multi_modal_projector is not None:
                logger.info("Adding multi_modal_projector to model parts")
                model_parts.append(self.multi_modal_projector)
            # Handle cases where lm_head exists at model level rather than language_model level
            if (
                getattr(self.language_model, "lm_head", None) is None
                and getattr(self.model, "lm_head", None) is not None
            ):
                logger.info("Adding lm_head to model parts")
                model_parts.append(self.model.lm_head)
            return model_parts
        else:
            return [self.language_model]

    @cached_property
    def _get_nparams_and_flops_fn(
        self, is_vision_model: bool
    ) -> Callable[[int], tuple[int, int]]:
        if is_vision_model:
            nparams = sum(p.numel() for p in self.parameters())
            nparams_embedding = sum(
                sum(p.numel() for p in m.parameters())
                for m in self.children()
                if isinstance(m, nn.Embedding)
            )

            # Reasoning behind the factor of 12 for the self-attention part of the formula:
            # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
            # 2. the flash attention does 1 more matmul recomputation in the backward
            #    but recomputation should not be counted in calculating MFU           (+0)
            # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
            # 4. we follow the convention and do not account for sparsity in causal attention
            num_heads = 0
            if hasattr(self.vision_config, "num_attention_heads"):
                num_heads = self.vision_config.num_attention_heads
            elif hasattr(self.vision_config, "num_heads"):
                num_heads = self.vision_config.num_heads
            else:
                logger.warning(f"Can not get num of heads from {self.vision_config}.")
            layers, heads, head_dim = (
                self.n_vision_layers,
                num_heads,
                self.vision_config.hidden_size // num_heads,
            )
            return lambda seq_len: (
                nparams,
                6 * (nparams - nparams_embedding)
                + 12 * layers * heads * head_dim * seq_len,
            )
        else:
            nparams = sum(p.numel() for p in self.parameters())
            nparams_embedding = sum(
                sum(p.numel() for p in m.parameters())
                for m in self.children()
                if isinstance(m, nn.Embedding)
            )
            # Reasoning behind the factor of 12 for the self-attention part of the formula:
            # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
            # 2. the flash attention does 1 more matmul recomputation in the backward
            #    but recomputation should not be counted in calculating MFU           (+0)
            # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
            # 4. we follow the convention and do not account for sparsity in causal attention
            layers, heads, head_dim = (
                self.n_lm_layers,
                self.text_config.num_attention_heads,
                self.text_config.hidden_size // self.text_config.num_attention_heads,
            )
            return lambda seq_len: (
                nparams,
                6 * (nparams - nparams_embedding)
                + 12 * layers * heads * head_dim * seq_len,
            )

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        n_params = 0
        n_flops = 0
        if self.vision_model is not None:
            n_params, n_flops = self._get_nparams_and_flops_fn(is_vision_model=True)(
                seq_len
            )

        lm_n_params, lm_n_flops = self._get_nparams_and_flops_fn(is_vision_model=False)(
            seq_len
        )
        n_params += lm_n_params
        n_flops += lm_n_flops
        return n_params, n_flops

    @classmethod
    def from_model_args(cls, hf_config) -> "HFModel":
        """
        Initialize a HFModel model from a HFModelArgs object.

        Args:
            hf_config : hf model config.

        Returns:
            HFModel: HFModel model.

        """
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        model_class = None
        quantization_config = getattr(hf_config, "quantization_config", None)
        need_dequantization = False
        if quantization_config is not None:
            if quantization_config["quant_method"] in ["mxfp4"]:
                assert hasattr(
                    transformers_quantization_config, "Mxfp4Config"
                ), "Mxfp4Config is not supported in this version of transformers. Please upgrade transformers to version 4.45.0 or higher."
                logger.warning(
                    "We don't support mxfp4 training for HFModel currently, will default to dequantizing the model to bf16/fp16."
                )
                need_dequantization = True
            hf_config.quantization_config["dequantize"] = need_dequantization

        pre_hf_models_patch(hf_config)

        try:
            model_class = load_model_class_by_config(hf_config)
            model = model_class(hf_config)
        except Exception as e:
            logger.warning(
                f"Got error({e}) when loading {hf_config.model_type}, Using AutoModel instead."
            )
            model_class = AutoModel
            model = AutoModel.from_config(hf_config, trust_remote_code=True)

        post_hf_models_patch(hf_config, model)

        return cls(
            hf_config,
            model,
            model_class,
            is_vlm=is_vlm,
            need_dequantization=need_dequantization,
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "HFModel":
        """
        Initialize a HFModel model from a pretrained model.

        Args:
            hf_config (AutoConfig): HuggingFace config.
            model_name_or_path (str): Model name or path to the pretrained model.
            max_position_embeddings (int): Maximum position embeddings.

        Returns:
            HFModel: HFModel model.

        """

        if max_position_embeddings is not None:
            hf_config.max_position_embeddings = max_position_embeddings
            if hasattr(hf_config, "text_config") and hasattr(
                hf_config.text_config, "max_position_embeddings"
            ):
                hf_config.text_config.max_position_embeddings = max_position_embeddings
            elif hasattr(hf_config, "llm_config") and hasattr(
                hf_config.llm_config, "max_position_embeddings"
            ):
                hf_config.llm_config.max_position_embeddings = max_position_embeddings

        return cls.from_model_args(hf_config)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    @classmethod
    def fqn_filter_for_fp8(cls) -> List[str]:
        llm = [
            "lm_head",
        ]
        visual = [
            "visual",
            "vision_tower",
        ]  # Filter Linear in visual out, they will corrupt the FP8 Linear.
        return llm + visual

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        assert cp_size == 1, "cp is not supported for HFModel"
        assert tp_size == 1, "tp is not supported for HFModel"

    def post_transform_of_local_view(self, local_view: torch.Tensor, name: str):
        if "gpt_oss" in self.hf_config.model_type:
            if "bias" not in name:  # Bias parameters do not require transposition
                if "gate_up_proj" in name or "down_proj" in name:

                    def transform(view):
                        return view.transpose(-2, -1).contiguous()

                    return partial(transform, local_view)

        return local_view

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
import copy

from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch

import vllm
import torch
from typing import List, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoConfig
from transformers import GenerationConfig
from vllm.entrypoints.llm import LLM
from vllm import SamplingParams
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import RolloutConfig
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.tools.tools_use import ToolParser
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
    add_tool_response_messages,
    add_assistant_message,
)
from cosmos_rl.tools.tools_use import OpenAIFunctionToolSchema


def vllm_version_check(rollout_config: RolloutConfig):
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )


def update_conversation_wth_rollout_result(
    conversation: ConversationType,
    rollout_result: str,
    tool_parser: ToolParser,
    tools: list[OpenAIFunctionToolSchema],
) -> ConversationType:
    """
    Update the conversation with the rollout result.

    1. if the rollout result contains tool calls, add the tool response messages to the conversation
    2. otherwise, add the assistant message to the conversation
    """
    content, function_calls = tool_parser.extract_tool_calls(rollout_result)

    if function_calls:
        conversation = add_tool_response_messages(conversation, function_calls)
    else:
        conversation = add_assistant_message(conversation, content)

    return conversation


class vLLMRollout(RolloutBase):
    def __init__(self, config: Config, tokenizer: AutoTokenizer, **kwargs):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            tokenizer: Tokenizer of the model.
            hf_config_path: huggingface config file path.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__()

        self.config = config
        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(model_path)

        self.pad_token_id = tokenizer.pad_token_id

        hf_config_path = self.config.policy.model_name_or_path
        try:
            generation_config = util.retry(GenerationConfig.from_pretrained)(
                hf_config_path
            )
            self.eos_token_ids = generation_config.eos_token_id
            if isinstance(self.eos_token_ids, int):
                self.eos_token_ids = [self.eos_token_ids]
        except Exception as e:
            logger.warning(
                f"[Rollout] Failed to load generation config from {hf_config_path}: {str(e)}, use default eos_token_id."
            )
            # self.eos_token_ids = [tokenizer.eos_token_id]
            # TODO(lms): remove this
            self.eos_token_ids = [151645, 151643]
        self.tokenizer = tokenizer
        self._engine_initialized = False
        self.rollout_engine = None

        self._model_param_map = None  # key: compatible name, value: param
        self.is_vlm = getattr(self.model_config, "vision_config", None) is not None

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        if not self._engine_initialized:
            trust_remote_code = True  # set trust remote code default to True.

            model_path = self.config.policy.model_name_or_path

            rollout_parallelism = self.rollout_config.parallelism

            # disable VLLM_DISABLE_COMPILE_CACHE
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

            tp_size = rollout_parallelism.tp_size
            pp_size = rollout_parallelism.pp_size

            enable_ep_parallelism = False
            disable_mm_preprocessor_cache = False

            # Check if the model has MoE
            moe_model_type = {"qwen3_moe"}
            multimodal_type = {"qwen2_5_vl"}

            model_type = self.model_config.model_type
            if model_type in moe_model_type:
                enable_ep_parallelism = True
            if model_type in multimodal_type:
                # for vllm nightly, this is only True for multimodal models, check here
                disable_mm_preprocessor_cache = True
            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size, but got tp_size: %d, pp_size: %d, world_size: %d"
                % (tp_size, pp_size, rollout_parallelism.world_size)
            )

            self.quantization = quantization

            policy_config = self.config.policy

            self.rollout_engine = LLM(
                model=model_path,
                enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                enable_expert_parallel=enable_ep_parallelism,
                distributed_executor_backend="external_launcher",
                dtype="auto",
                enforce_eager=self.rollout_config.enforce_eager,  # enable cuda graph
                gpu_memory_utilization=self.rollout_config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
                skip_tokenizer_init=False,
                max_model_len=policy_config.model_max_length,
                disable_log_stats=True,
                # default to 2048, this is related with chunked prefill. https://docs.vllm.ai/en/latest/performance/optimization.html
                max_num_batched_tokens=2048
                if 2048 >= policy_config.model_max_length
                else policy_config.model_max_length,
                enable_chunked_prefill=self.rollout_config.enable_chunked_prefill,
                # Always disable prefix caching, since RL will change the underlying model.
                # The prefix cache will be invalid after training.
                enable_prefix_caching=False,
                trust_remote_code=trust_remote_code,
                quantization=self.quantization,
                seed=seed or 42,
                load_format=load_format,
            )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            # initialization done.

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                from vllm.config import set_current_vllm_config

                vllm_config = self.rollout_engine.llm_engine.vllm_config
                with set_current_vllm_config(vllm_config):
                    apply_fp8_linear_patch(self.get_underlying_model())

    @torch.no_grad()
    def rollout_generation_single_turn(
        self,
        prompt_id_and_payload_list: List[Tuple[int, Any]],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[List[str]]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        # List of payloads.
        # [
        #   payload,
        #   payload,
        #   ...
        # ]
        payloads = [x[1] for x in prompt_id_and_payload_list]

        # Pack the payloads into prompts for vllm.
        prompts = [data_packer.get_rollout_input(payload) for payload in payloads]
        prompts = data_packer.rollout_collate_fn(prompts)
        if self.is_vlm:
            new_prompts = util.decode_vision_info(prompts)
        else:
            new_prompts = prompts

        # List of completions per prompt.
        # [
        #   [completion_str, completion_str, ...],
        #   [completion_str, completion_str, ...],
        #   ...
        # ]
        response: List[List[str]] = []

        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            with torch.cuda.stream(stream):
                results = self.rollout_engine.generate(
                    new_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            for output in results:
                response.append(
                    [output.outputs[i].text for i in range(len(output.outputs))]
                )
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        return response

    @torch.no_grad()
    def rollout_generation_multi_turn(
        self,
        prompt_id_and_payload_list: List[Tuple[int, Any]],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[Any]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        stream = torch.cuda.current_stream() if stream is None else stream

        def generation_multi_turn_for_one_payload(payload: Any) -> Any:
            assistant_turn_count = 0
            while (
                assistant_turn_count
                < self.rollout_config.multi_turn_config.max_assistant_turns
            ):
                # Pack the payloads into prompts for vllm.
                prompts = [data_packer.get_rollout_input(payload)]
                prompts = data_packer.rollout_collate_fn(prompts)

                with torch.cuda.stream(stream):
                    results = self.rollout_engine.generate(
                        prompts=prompts,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )

                payload = data_packer.extend_conversation(
                    payload, results[0].outputs[0].text
                )

                # check if the sequence length is reached the max_sequence_length
                if (
                    len(results[0].prompt_token_ids)
                    + len(results[0].outputs[0].token_ids)
                    > self.rollout_config.max_response_length
                ):
                    logger.warning(
                        "[Rollout] The sequence length is reached the max_response_length, stop the multi-turn generation."
                    )
                    break

                assistant_turn_count += 1

            return payload

        # List of completions per prompt.
        # [
        #   [completion_str, completion_str, ...],
        #   [completion_str, completion_str, ...],
        #   ...
        # ]
        response: List[Any] = []
        for payload_list in prompt_id_and_payload_list:
            response.append(
                generation_multi_turn_for_one_payload(copy.deepcopy(payload_list[0]))
            )

        return response

    def rollout_generation(
        self,
        prompt_id_and_payload_list: List[Tuple[int, Any]],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[List[str]]:
        if self.rollout_config.multi_turn_config.enable:
            return self.rollout_generation_multi_turn(
                prompt_id_and_payload_list, stream, data_packer, sampling_params
            )
        else:
            return self.rollout_generation_single_turn(
                prompt_id_and_payload_list, stream, data_packer, sampling_params
            )

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine

    def is_engine_initialized(self):
        return self._engine_initialized

    def fp8_quantization(self, weight: torch.Tensor):
        # convert to fp8
        from vllm import _custom_ops as ops

        # quantization of rowwise torch scaled_mm.
        # weight has shape [out_dim, in_dim]
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight, scale=None, use_per_token_if_dynamic=True
        )

        return qweight.t(), weight_scale

    def model_param_map(self, weight_mapper: WeightMapper):
        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.named_parameters():
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(name)
            param_map[compatible_name] = param
        self._model_param_map = param_map
        return self._model_param_map

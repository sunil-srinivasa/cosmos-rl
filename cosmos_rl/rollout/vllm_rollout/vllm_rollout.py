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
import vllm
import torch
from typing import List, Tuple, Any
from transformers import AutoTokenizer, AutoConfig
from transformers import GenerationConfig
from vllm.entrypoints.llm import LLM
from vllm import SamplingParams
from functools import partial
import types
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.rollout.vllm_rollout.vllm_patch import (
    patch_vllm_model_to_reload_weight,
)
from cosmos_rl.utils.constant import (
    COSMOS_ROLLOUT_STEP_INTERVAL,
)
from cosmos_rl.policy.config import RolloutConfig
from cosmos_rl.dispatcher.data.packer import DataPacker

from cosmos_rl.dispatcher.command import Command, RolloutToRolloutBroadcastCommand


def vllm_version_check(rollout_config: RolloutConfig):
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )


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

        trust_remote_code = True  # set trust remote code default to True.
        model_path = policy_config.model_name_or_path

        # Check if the model has MoE
        model_config = util.retry(AutoConfig.from_pretrained)(
            model_path, trust_remote_code=trust_remote_code
        )

        enable_ep_parallelism = False
        disable_mm_preprocessor_cache = False

        moe_model_type = {"qwen3_moe"}
        multimodal_type = {"qwen2_5_vl"}

        model_type = model_config.model_type
        if model_type in moe_model_type:
            enable_ep_parallelism = True
        if model_type in multimodal_type:
            # for vllm nightly, this is only True for multimodal models, check here
            disable_mm_preprocessor_cache = True

        rollout_parallelism = self.rollout_config.parallelism

        tp_size = rollout_parallelism.tp_size
        pp_size = rollout_parallelism.pp_size

        assert (
            tp_size * pp_size == rollout_parallelism.world_size
        ), "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size."

        # disable VLLM_DISABLE_COMPILE_CACHE
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        # if flashinfer config is not enabled, avoid importing flashinfer
        if self.config.rollout.vllm_use_flashinfer:
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                logger.warning(
                    "[Rollout] flashinfer is not installed, ignore rollout.vllm_use_flashinfer setting."
                )
            else:
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        if self.config.rollout.sampling_config.use_flashinfer:
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                logger.warning(
                    "[Rollout] flashinfer is not installed, ignore rollout.sampling_config.use_flashinfer setting."
                )
            else:
                os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"

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
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=kwargs.get("seed") or 42,
            # Note: We set load_format="dummy" to avoid loading the HF model weights which could cause too many requests from multiple replicas.
            # This will affect:
            #      1. for the benchmark, the result won't be correct because now we have random weights. But it is fine for profiling.
            #      2. TODO:(lms) this may have conflict with quantization. Check it when supporting quantization
            # This won't affect:
            #      1. The GRPO procedure won't be affected because we will first have a P2R before rollout generation. So it is safe.
            load_format=kwargs.get("load_format", "dummy"),
        )

        # patch the vllm model to reload weight
        patch_vllm_model_to_reload_weight(self.rollout_engine)

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

        # sampling params
        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p,
            top_k=self.config.validation.top_k,
            temperature=self.config.validation.temperature,
            repetition_penalty=self.config.validation.repetition_penalty,
            max_tokens=self.config.validation.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )

        self.sampling_params = SamplingParams(
            n=self.config.rollout.n_generation,
            logprobs=0,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )

    def reload_weight(self):
        self.rollout_engine.llm_engine.vllm_config.load_config.load_format = "auto"
        self.rollout_engine.collective_rpc("reload_model")

    @torch.no_grad()
    def rollout_generation(
        self,
        prompt_id_and_payload_list: List[Tuple[int, Any]],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
    ) -> List[List[str]]:
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
                    prompts=prompts,
                    sampling_params=self.sampling_params,
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
            return None

        return response

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    @classmethod
    def patch_vllm_rollout_locked_step(
        cls, rollout, consume_command, enable_validation
    ):
        """
        Patch the vLLM rollout locked step to support validation.
        """
        llm_engine = rollout.rollout_engine.llm_engine
        orig_step = llm_engine.step

        def cmd_pred(cmd: Command, enable_validation: bool):
            if enable_validation and isinstance(cmd, RolloutToRolloutBroadcastCommand):
                return False
            return True

        def step(self, *args, **kwargs):
            if not hasattr(self, "_cosmos_step_counter"):
                self._cosmos_step_counter = 0
            self._cosmos_step_counter += 1
            if self._cosmos_step_counter % COSMOS_ROLLOUT_STEP_INTERVAL == 0:
                # IMPORTANT:
                # If validation is enabled, R2R is not expected to be called in this step function
                # to avoid recursive inference execution.
                consume_command(
                    cmd_pred=partial(cmd_pred, enable_validation=enable_validation)
                )
            return orig_step(*args, **kwargs)

        llm_engine.step = types.MethodType(step, llm_engine)

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

import torch
import threading
from queue import Queue
import atexit
import types
from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from typing import List, Optional, Callable, Union
from functools import partial
from transformers import AutoConfig
from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.constant import (
    COSMOS_ROLLOUT_STEP_INTERVAL,
    COSMOS_ROLLOUT_REPORT_INTERVAL,
)
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    Command,
)
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_broadcast,
    nccl_recv,
    nccl_group_start,
    nccl_group_end,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    WeightSyncInstructionsGroup,
)
import cosmos_rl.utils.distributed as dist_util
import cosmos_rl.utils.util as util
from cosmos_rl.utils import constant
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    IdxAndRLPayload,
    ConversationType,
)
from cosmos_rl.rollout.schema import RolloutResult
from torch.utils.data import Dataset
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from vllm import SamplingParams
import time

"""
Keep in mind that torch distributed is not thread safe. So try to keep the usage in the same thread.
"""


def _patch_vllm_rollout_locked_step(
    rollout: vLLMRollout, consume_command, reward_fetch, enable_validation
):
    llm_engine = rollout.get_engine().llm_engine
    orig_step = llm_engine.step

    def cmd_pred(cmd: Command, enable_validation: threading.Event):
        # Make sure no weight update happens during validation.
        # So filter out R2R and P2R commands when validation is enabled.
        if enable_validation.is_set() and (
            isinstance(cmd, RolloutToRolloutBroadcastCommand)
            or isinstance(cmd, PolicyToRolloutUnicastCommand)
        ):
            return False
        return True

    def step(self, *args, **kwargs):
        if not hasattr(self, "_cosmos_step_counter"):
            self._cosmos_step_counter = 0
        self._cosmos_step_counter += 1

        if (
            COSMOS_ROLLOUT_REPORT_INTERVAL > 0
            and self._cosmos_step_counter % COSMOS_ROLLOUT_REPORT_INTERVAL == 0
        ):
            _, is_validation, _, _ = reward_fetch()
            assert not is_validation, "Validation report should be handled in the broadcast command rather than step function."

        if (
            COSMOS_ROLLOUT_STEP_INTERVAL > 0
            and self._cosmos_step_counter % COSMOS_ROLLOUT_STEP_INTERVAL == 0
        ):
            # IMPORTANT:
            # If validation is enabled, R2R is not expected to be called in this step function
            # to avoid recursive inference execution.
            consume_command(
                cmd_pred=partial(cmd_pred, enable_validation=enable_validation)
            )
        return orig_step(*args, **kwargs)

    llm_engine.step = types.MethodType(step, llm_engine)


class vLLMRolloutWorker(RolloutWorkerBase):
    """
    vLLMRolloutWorker will be a replica instance of single DP.
    vLLMRolloutWorker should support scaling launch.
    """

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super(vLLMRolloutWorker, self).__init__(config, parallel_dims)

        self.state = State()

        if self.config.rollout.parallelism.dp_shard_size == -1:
            self.config.rollout.parallelism.dp_shard_size = parallel_dims.dp_shard
        assert self.config.rollout.parallelism.dp_shard_size == parallel_dims.dp_shard
        assert (
            self.config.rollout.parallelism.dp_shard_size > 0
        ), "[Rollout] dp_shard_size should be greater than 0."

        # CommandQueue queried from controller.
        self._command_queue: Queue[Command] = Queue()
        self._prompt_queue: Queue[List[IdxAndRLPayload]] = Queue()
        self.current_weight_version = 0

        # determine the quantization type
        self.quantization_type = None
        if self.config.rollout.quantization != "none":
            self.quantization_type = self.config.rollout.quantization

        self.rollout: vLLMRollout = vLLMRollout(self.config, self.tokenizer)

        # communicator index for the cached communicators in C++ binding.
        self.global_commnicator_idex = -1
        # rank in current rollout replicas.
        self.rank_in_rollout_repicas = -1

        # cache for NCCL communicators for P2R.
        self.policy_to_rollout_nccl_communicators = {}

        self.batch_size = self.config.rollout.batch_size
        if self.config.validation.enable:
            self.val_batch_size = self.config.validation.batch_size or self.batch_size
            assert (
                self.val_batch_size > 0
            ), "[Rollout] val_batch_size should be greater than 0."
        else:
            self.val_batch_size = None
        self.background_thread: threading.Thread | None = None

        # For Polocy to Rollout weight mapping
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        model_type = hf_config.model_type
        if self.quantization_type == "mxfp4":
            assert (
                model_type == "gpt_oss"
            ), "[Rollout] Mxfp4 quantization is only supported for GPT-OSS now."

        if not ModelRegistry.check_model_type_supported(model_type):
            logger.warning(
                f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
            )
            model_type = constant.COSMOS_HF_MODEL_TYPES
        self.weight_mapper = WeightMapper.get_weight_mapper(model_type)(hf_config)
        self.model_config = hf_config

        atexit.register(self.handle_shutdown)

        self.inference_stream = torch.cuda.Stream()

        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.rollout.eos_token_ids,
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
            stop_token_ids=self.rollout.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )

        # Holding temp tensors created in `recv_tensor_creator`. Do not remove this, or
        self.total_temp_tensor_pool = []
        self.misc_params = set()
        self.validation_flag = threading.Event()
        self.reward_dispatcher = RewardDispatcher()

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        num_workers: int = 8,
    ):
        self.reward_dispatcher.setup(
            config=self.config,
            dataset=dataset,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            num_workers=num_workers
            if self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            else 0,
        )

    def prepare_shard_infos_for_weight_sync_insts(self):
        if self.quantization_type == "fp8":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                post_process_view_map_for_fp8 as post_process_view_map_for_lowp,
            )
        elif self.quantization_type == "mxfp4":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )

            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                post_process_view_map_for_mxfp4 as post_process_view_map_for_lowp,
            )

        if self.quantization_type is not None:
            promotion_dtype = util.str2torch_dtype(self.config.train.param_dtype)
            self.vllm_hp_weight_map, self.vllm_quantized_weight_map = (
                cache_weight_of_quantized_module(
                    self.get_underlying_model(),
                    promotion_dtype,
                    self.weight_mapper,
                    self.parallel_dims,
                )
            )
            # replace the weight of quantized module with the high precision weight.
            # let weight in vllm_weight_inplace_view_map always in high precision for recv
            # high precision weight from policy.
            replace_weight_of_quantized_module(
                self.get_underlying_model(),
                self.vllm_hp_weight_map,
                self.weight_mapper,
            )

        self.vllm_weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.cosmos_rollout_prepare_recv(self.get_underlying_model())
        )
        self.recv_param_key_n_rank_list = []
        param_groups = []
        for group in grouped_recv_param_key_n_rank_list:
            self.recv_param_key_n_rank_list.extend(group)
            if len(group) > 1:
                param_groups.append([x[0] for x in group])
        self.recv_param_key_n_rank_list = sorted(
            self.recv_param_key_n_rank_list, key=lambda x: x[0]
        )
        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)

        # this must be done after prepare_local_shard_infos
        if self.quantization_type is not None:
            self.vllm_weight_inplace_view_map = post_process_view_map_for_lowp(
                self.vllm_weight_inplace_view_map
            )
            # Get vllm weight back into quantized.
            replace_weight_of_quantized_module(
                self.get_underlying_model(),
                self.vllm_quantized_weight_map,
                self.weight_mapper,
            )

        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        all_param_groups = dist_util.all_gather_object_cpu(param_groups)
        merged_groups = {}
        for r, param_groups in enumerate(all_param_groups):
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) != 0:
                continue
            for group in param_groups:
                group = sorted(group)
                key = tuple(group)
                if key not in merged_groups:
                    merged_groups[key] = group
        sorted_params_all_rank = dist_util.all_gather_object_cpu(
            [x[0] for x in self.recv_param_key_n_rank_list]
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        if self.global_rank == 0:
            self.api_client.post_rollout_shard_info(
                shard_infos=self.all_rank_local_shard_infos,
                param_groups=list(merged_groups.values()),
                sorted_params=sorted_params_all_rank,
            )

    def handle_shutdown(self):
        # Only call once
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True
            if not self.shutdown_signal.is_set():
                logger.info(
                    f"[Rollout] shutdown instruction of {self.replica_name}, setting shutdown signal"
                )
                self.shutdown_signal.set()
            if not self.shutdown_mp_signal.is_set():
                self.shutdown_mp_signal.set()
            if self.background_thread is not None:
                self.background_thread.join()
                self.background_thread = None

            if self.heartbeat_thread is not None:
                self.heartbeat_thread.join()
                self.heartbeat_thread = None
            self.unregister_from_controller()

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return self.rollout.get_underlying_model()

    @RolloutWorkerBase.register_rollout_command_handler(BuildMeshCommand)
    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        logger.info(f"[Rollout] Building global mesh for {self.replica_name}")

        replica_name_to_rank = build_mesh_command.replica_name_to_rank
        if self.replica_name not in replica_name_to_rank:
            raise RuntimeError(
                f"[Rollout] Replica {self.replica_name} not found in registered replicas."
            )
        self.rank_in_rollout_repicas = replica_name_to_rank[self.replica_name]

        if len(replica_name_to_rank) == 1:
            # only one rollout replica now, no need to build mesh.
            return
        # generate key for storing the NCCL group id.
        # group_0: [rank 0 in replica 0, rank 0 in replica 1, ..., rank 0 in replica n-1]
        # group_1: [rank 1 in replica 0, rank 1 in replica 1, ..., rank 1 in replica n-1]
        # ...
        # group_m-1: [rank m-1 in replica 0, rank m-1 in replica 1, ..., rank m-1 in replica n-1]
        unique_rollout_group_key = self.get_group_unique_key(replica_name_to_rank)
        nccl_group_id = None
        if self.rank_in_rollout_repicas == 0:
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = create_nccl_uid()
            self.api_client.post_nccl_comm_initiator(
                unique_rollout_group_key, nccl_group_id
            )

        if self.rank_in_rollout_repicas != 0:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )

        # update the cached communicator index
        logger.debug(
            f"[Rollout] Creating nccl communicator for global mesh: {unique_rollout_group_key}"
        )
        self.global_commnicator_idex = create_nccl_comm(
            nccl_group_id, self.rank_in_rollout_repicas, len(replica_name_to_rank)
        )
        # update the replcia_name to rank dict
        self.replica_name_to_rank = replica_name_to_rank

    def query_nccl_unique_id_from_controller(self, unique_id_key: str):
        # We don't have something like dist.barrier(), so just use while True loop to query it like synchronize.
        # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
        return self.api_client.post_nccl_comm_acceptor(unique_id_key)

    def prepare_trainable_params(self):
        # TODO: (lms/feng) Refactor the param management logic for P2R and R2R, incluing trainable params for P2R and non-trainable params for R2R.
        if not hasattr(self, "trainable_params"):
            if self.global_rank == 0:
                trainable_params = self.api_client.get_trainable_params()
            else:
                trainable_params = None
            trainable_params = dist_utils.broadcast_object_cpu(
                trainable_params,
            )
            self.trainable_params = set(trainable_params)
            logger.info(
                f"[Rollout] Finished fetching {len(self.trainable_params)} trainable params from controller."
            )
            # The splitted and unsplited version of param names should both added to handle for P2R and R2R cases separately.
            for p in trainable_params:
                self.trainable_params.add(
                    self.weight_mapper.get_unsplited_weight_name(p)
                )

            # Add weight scale of quantized weights to trainable params
            if self.quantization_type is not None:
                # Trivial params:
                # including tensors that need to be synced but not trainable in R2R. These
                # tensors will not be synced from P2R, so we have to add them to trainable params.
                for name, _ in self.rollout.model_param_map(self.weight_mapper).items():
                    if name.endswith("_scale"):
                        self.misc_params.add(name)
                self.trainable_params.update(self.misc_params)

            logger.info(
                f"[Rollout] Obtained {len(self.trainable_params)} trainable params after weight unsplit."
            )

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_group: WeightSyncInstructionsGroup,
        communicator_index: int,
        trainable_only: bool,
        do_weight_sync_check: bool = False,
    ):
        target_dtype = str2torch_dtype(self.config.train.transfer_dtype)
        check_inside_group = do_weight_sync_check
        if self.quantization_type is not None:
            inst_group_weight_name = (
                insts_group.param_instructions[0].param_name
            )  # take a name from the inst group to determine the full weight name
            # the full weight name that this inst group handles.
            inst_group_full_weight_name = self.weight_mapper.get_unsplited_weight_name(
                inst_group_weight_name
            )
            is_lowp_quantized_module = (
                inst_group_full_weight_name in self.vllm_quantized_weight_map
            )
            check_inside_group = do_weight_sync_check and (not is_lowp_quantized_module)

        total_bytes_received = 0

        all_tensor_views_to_copy = []
        tensors_to_check = []

        def recv_tensor_creator(vllm_tensor_view: torch.Tensor):
            recv_tensor = None
            inplace = True

            if vllm_tensor_view.is_contiguous():
                recv_tensor = vllm_tensor_view
            else:
                # new a temp tensor
                recv_tensor = torch.empty_like(vllm_tensor_view).contiguous()
                inplace = False

            if vllm_tensor_view.dtype != target_dtype:
                recv_tensor = recv_tensor.to(target_dtype)
                inplace = False
            # Hold these recv_tensor, in case of buffer reusing by torch
            self.total_temp_tensor_pool.append(recv_tensor)

            return recv_tensor, inplace

        skipped_params_cnt = 0

        for insts_for_per_param in insts_group.param_instructions:
            # insts_for_per_param: WeightSyncInstructionsPerParam -> inst collection for a single tensor
            insts = insts_for_per_param.instructions
            # insts: List[Tuple[int, int, Dict[int, Any]]]
            inst_dest_name = insts_for_per_param.param_name

            if inst_dest_name not in self.trainable_params and trainable_only:
                logger.info(
                    f"[Rollout] Skip {inst_dest_name} in P2R recv due to non trainable."
                )
                skipped_params_cnt += 1
                continue

            target_tensor = self.vllm_weight_inplace_view_map[inst_dest_name]

            if check_inside_group:
                cloned_target_tensor = target_tensor.clone()
                # clear the current view
                target_tensor.zero_()

            for inst in insts:
                # Inst for different part of a tensor between policy and rollout.
                p_rank = inst.policy_rank
                r_rank = inst.rollout_rank
                tensor_split_strategys = inst.slice_strategy
                assert r_rank == global_rank_of_rollout
                vllm_tensor_view = target_tensor.cosmos_slice(tensor_split_strategys)
                recv_tensor, inplace = recv_tensor_creator(vllm_tensor_view)
                logger.debug(
                    f"[Rollout] Recving tensor {inst_dest_name} from policy rank {p_rank} to rollout rank {r_rank}, shape {vllm_tensor_view.shape} of {target_tensor.shape} with dtype {vllm_tensor_view.dtype}."
                )
                nccl_recv(recv_tensor, p_rank, communicator_index)

                # inplace copy
                if not inplace:
                    all_tensor_views_to_copy.append(
                        (vllm_tensor_view, recv_tensor, inst_dest_name)
                    )

                total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()

            if check_inside_group:
                tensors_to_check.append(
                    (cloned_target_tensor, target_tensor, insts, inst_dest_name)
                )

        post_process_list_for_lowp = []

        if not check_inside_group and self.quantization_type is not None:
            post_process_list_for_lowp.append(inst_group_full_weight_name)

        def completion_lambda(
            all_tensor_views_to_copy, tensors_to_check, post_process_list_for_lowp
        ):
            for view, recv_tensor, inst_dest_name in all_tensor_views_to_copy:
                self.weight_mapper.update_tensor_view(
                    view, recv_tensor, inst_dest_name, parallel_dims=self.parallel_dims
                )

            for (
                cloned_target_tensor,
                target_tensor,
                insts,
                inst_dest_name,
            ) in tensors_to_check:
                cloned_target_tensor = cloned_target_tensor.to(target_dtype).to(
                    cloned_target_tensor.dtype
                )
                if not torch.allclose(cloned_target_tensor, target_tensor):
                    raise ValueError(
                        f"Weight sync check failed after weight sync instruction: {insts} for {inst_dest_name}."
                    )
            tensors_to_check.clear()

            # here we got one full weight tensor sync done, if it is fp8/mxfp4 weight, we should do the quantization and check the numerical error.
            if self.quantization_type is not None:
                for inst_group_full_weight_name in post_process_list_for_lowp:
                    if self.quantization_type == "fp8":
                        if inst_group_full_weight_name in self.vllm_hp_weight_map:
                            weight_to_quantize = self.vllm_hp_weight_map[
                                inst_group_full_weight_name
                            ]  # [out_dim, in_dim]
                            quantized_weight, weight_scale = (
                                self.rollout.fp8_quantization(weight_to_quantize)
                            )
                            model_param_map = self.rollout.model_param_map(
                                self.weight_mapper
                            )
                            vllm_native_weight = model_param_map[
                                inst_group_full_weight_name
                            ]

                            # check weight sync
                            if do_weight_sync_check:
                                # allclose doesn't support fp8, promote it.
                                bf16_vllm_native_weight = vllm_native_weight.to(
                                    torch.bfloat16
                                )
                                bf16_quantized_weight = quantized_weight.to(
                                    torch.bfloat16
                                )
                                if not torch.allclose(
                                    bf16_vllm_native_weight, bf16_quantized_weight
                                ):
                                    raise ValueError(
                                        f"FP8 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                    )
                            vllm_native_weight.copy_(quantized_weight)
                            # get the scale key.
                            scale_key = inst_group_full_weight_name.replace(
                                ".weight", ".weight_scale"
                            )
                            scale_tensor = model_param_map[scale_key]
                            assert (
                                scale_tensor.shape == weight_scale.shape
                            ), f"scale_tensor.shape: {scale_tensor.shape}, weight_scale.shape: {weight_scale.shape}"
                            scale_tensor.copy_(weight_scale)
                    elif self.quantization_type == "mxfp4":
                        # Note: For mxfp4, we don't do weight sync check for quantized weights.
                        if inst_group_full_weight_name in self.vllm_hp_weight_map:
                            if "gate_up_proj_bias" not in inst_group_full_weight_name:
                                # Weight to quantize:
                                # [local_num_experts, 2* local_intermediate_size, hidden_size] for gate_up_proj
                                # [local_num_experts, hidden_size, local_intermediate_size] for down_proj
                                weight_to_quantize = self.vllm_hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                quantized_weight, weight_scale = (
                                    self.rollout.mxfp4_quantization(weight_to_quantize)
                                )
                                # The quantized version of the weight has been removed by vLLM internally.
                                # https://github.com/zyongye/vllm/blob/6a70830065701b163e36a86fd331b41b5feac401/vllm/model_executor/layers/quantization/mxfp4.py#L328
                                # We can't get it from named_parameters.
                                vllm_native_weight = None
                                vllm_native_weight_scale = None

                                for (
                                    module_name,
                                    module,
                                ) in self.get_underlying_model().named_modules():
                                    w13_weight_name = f"{module_name}.w13_weight"
                                    w2_weight_name = f"{module_name}.w2_weight"
                                    w13_compatible_weight_name = (
                                        self.weight_mapper._rollout_vllm_name_to_hf(
                                            w13_weight_name
                                        )
                                    )
                                    w2_compatible_weight_name = (
                                        self.weight_mapper._rollout_vllm_name_to_hf(
                                            w2_weight_name
                                        )
                                    )

                                    # mxfp4 weight and mxfp4 weight scale are in int8 data type.
                                    # Two fp4 are packed into one int8 memory.
                                    if (
                                        inst_group_full_weight_name
                                        == w13_compatible_weight_name
                                    ):
                                        vllm_native_weight = module.quant_method.w13_weight_triton_tensor.storage.data
                                        vllm_native_weight_scale = module.quant_method.w13_precision_config.weight_scale.storage.data
                                        break
                                    elif (
                                        inst_group_full_weight_name
                                        == w2_compatible_weight_name
                                    ):
                                        vllm_native_weight = module.quant_method.w2_weight_triton_tensor.storage.data
                                        vllm_native_weight_scale = module.quant_method.w2_precision_config.weight_scale.storage.data
                                        break

                                assert (
                                    vllm_native_weight is not None
                                ), f"Failed to find the original weight for {inst_group_full_weight_name}"
                                assert (
                                    vllm_native_weight_scale is not None
                                ), f"Failed to find the original weight scale for {inst_group_full_weight_name}"

                                with torch.inference_mode():
                                    _, dim_1, dim_2 = quantized_weight.shape

                                    # check weight sync
                                    if do_weight_sync_check:
                                        valid_native_weight = vllm_native_weight[
                                            :, :dim_1, :dim_2
                                        ]
                                        if not torch.allclose(
                                            valid_native_weight, quantized_weight
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    vllm_native_weight[:, :dim_1, :dim_2].copy_(
                                        quantized_weight
                                    )
                                    # check weight sync
                                    _, dim_1, dim_2 = weight_scale.shape
                                    if do_weight_sync_check:
                                        valid_native_weight_scale = (
                                            vllm_native_weight_scale[:, :dim_1, :dim_2]
                                        )
                                        if not torch.allclose(
                                            valid_native_weight_scale, weight_scale
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight scale doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    vllm_native_weight_scale[:, :dim_1, :dim_2].copy_(
                                        weight_scale
                                    )

                            else:
                                # For w13_bias, no need to quant, just copy the weight.
                                w13_bias_hp_weight = self.vllm_hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                model_param_map = self.rollout.model_param_map(
                                    self.weight_mapper
                                )
                                vllm_native_weight = model_param_map[
                                    inst_group_full_weight_name
                                ]
                                _, dim1 = w13_bias_hp_weight.shape
                                if do_weight_sync_check:
                                    if not torch.allclose(
                                        vllm_native_weight[:, :dim1], w13_bias_hp_weight
                                    ):
                                        raise ValueError(
                                            f"gate_up_proj_bias doesn't match after weight sync for full weight name: {inst_group_full_weight_name}."
                                        )

                                vllm_native_weight[:, :dim1].copy_(w13_bias_hp_weight)
            else:
                # For non-fp8/mxfp4 weights and fp8/mxfp4 not enabled cases, we just do nothing
                pass

        return (
            total_bytes_received,
            partial(
                completion_lambda,
                all_tensor_views_to_copy,
                tensors_to_check,
                post_process_list_for_lowp,
            ),
            skipped_params_cnt,
        )

    def do_validation(self):
        validation_queue = Queue()
        prompt_idxs: List[int] = []
        validation_payloads: List[RLPayload] = []
        # Do validation here
        while True:
            is_end = self.request_new_prompts(
                self.val_batch_size,
                validation_queue,
                validation_step=self.current_step,
            )
            if not validation_queue.empty():
                prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                    validation_queue.get()
                )
                payloads = [p for _, p in prompt_id_and_payload_list]
                rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                    payloads=payloads,
                    stream=self.inference_stream,
                    data_packer=self.val_data_packer,
                    sampling_params=self.val_sampling_params,
                )
                if rollout_results:
                    prompt_idxs.extend([idx for idx, _ in prompt_id_and_payload_list])
                    for p, rr in zip(payloads, rollout_results):
                        p.completions = rr.completions
                        if self.rollout.rollout_config.multi_turn_config.enable:
                            p.completed_conversations = rr.completed_conversations
                    validation_payloads.extend(payloads)

            if is_end:
                break

        # Clear the flag to indicate validation is done.
        self.validation_flag.clear()
        should_report = self.parallel_dims.tp_coord[0] == 0 and (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        if should_report:
            self.reward_dispatcher.enqueue_rewards_cal(
                validation_payloads, True, self.current_step, prompt_idxs
            )
            payloads, is_validation, current_step, empty = self.report_rollouts(
                block=True
            )
            assert (
                (is_validation and payloads is not None or payloads is None)
                and (not empty or len(validation_payloads) == 0)
            ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
            while not empty:
                assert (
                    is_validation or payloads is None
                ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
                if payloads is not None:
                    response = ValidationReportRequest(
                        src_replica_name=self.replica_name,
                        validation_step=current_step,
                        prompt_idxs=[],
                        payloads=payloads,
                        is_end=True,
                    )
                    self.api_client.post_validation_report(response)
                payloads, is_validation, current_step, empty = (
                    self.reward_dispatcher.dequeue_rewards_cal()
                )

    def lazy_initialize_rollout_engine(self, load_format):
        # lazy initialization of the vllm engine.
        if not self.rollout.is_engine_initialized():
            self.rollout.init_engine(
                quantization=self.quantization_type,
                seed=self.config.rollout.seed,
                load_format=load_format,
            )
            _patch_vllm_rollout_locked_step(
                self.rollout,
                self.consume_command,
                self.report_rollouts,
                self.validation_flag,
            )
            self.prepare_shard_infos_for_weight_sync_insts()

    @RolloutWorkerBase.register_rollout_command_handler(PolicyToRolloutUnicastCommand)
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
        # lazy initialization of the vllm engine.
        is_for_weight_resume = command.dst_replica_name == self.replica_name
        load_format = "auto" if is_for_weight_resume else "dummy"
        self.lazy_initialize_rollout_engine(load_format)

        if command.dst_replica_name != self.replica_name:
            return
        # get the nccl_unique_id from the controller
        communicator_index = {}
        nccl_unique_id_key = command.src_replica_name + "_" + command.dst_replica_name
        if nccl_unique_id_key in self.policy_to_rollout_nccl_communicators:
            logger.debug(
                f"[Rollout] Reusing cached communicator for {nccl_unique_id_key}"
            )
            communicator_index = self.policy_to_rollout_nccl_communicators[
                nccl_unique_id_key
            ]
        else:
            logger.debug(f"[Rollout] Querying nccl group id for {nccl_unique_id_key}")
            # query the nccl group id from controller
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                nccl_unique_id_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )
            # create the communicator index
            # p_rank is the rank in policy, r_rank is the rank in rollout
            communicator_index = create_nccl_comm(
                nccl_group_id,
                self.global_rank + command.src_replica_size,
                self.world_size + command.src_replica_size,
            )
            # cache the communicator index
            self.policy_to_rollout_nccl_communicators[nccl_unique_id_key] = (
                communicator_index
            )

        if not hasattr(self, "policy_to_rollout_recv_insts"):
            assert (
                not command.trainable_only
            ), "all params must be transferred at the first time P2R"
            logger.info(
                "[Rollout] Fetching policy_to_rollout_recv_insts from controller ..."
            )
            self.policy_to_rollout_recv_insts = (
                self.api_client.post_rollout_shard_recv_insts(self.global_rank)
            )
            logger.info(
                "[Rollout] Finished policy_to_rollout_recv_insts from controller."
            )
        else:
            assert (
                command.trainable_only
            ), "only trainable params should be transferred at the not first time P2R"

        self.prepare_trainable_params()

        total_recvs = 0
        total_params = 0
        for insts_group in self.policy_to_rollout_recv_insts:
            for insts_for_per_param in insts_group.param_instructions:
                total_params += 1
                total_recvs += len(insts_for_per_param.instructions)

        copy_stream = torch.cuda.Stream()

        assert (
            total_params == len(self.recv_param_key_n_rank_list)
        ), f"Mismatch in total params and received param keys: {total_params} != {len(self.recv_param_key_n_rank_list)}"

        with torch.cuda.stream(self.inference_stream):
            logger.info(
                f"Starting to execute {len(self.policy_to_rollout_recv_insts)}; {total_params}, {total_recvs} weight sync receives ..."
            )
            # recv the weight from policy
            st = time.time()
            total_bytes_received = 0

            pending_bytes = [0]
            pending_completions = []
            pending_groups = 0

            def flush_completions(pending_bytes, pending_completions):
                recv_ready = torch.cuda.Event()
                recv_ready.record()
                with torch.cuda.stream(copy_stream):
                    recv_ready.wait()
                    logger.debug(
                        f"Flushing {len(pending_completions)} completions, {pending_bytes[0] // 1024 // 1024}"
                    )
                    for completion in pending_completions:
                        completion()
                    pending_bytes[0] = 0
                    pending_completions.clear()

            nccl_group_start(communicator_index)

            skipped_params_cnt = 0
            transferred_params_cnt = 0
            skipped_groups_cnt = 0
            transferred_groups_cnt = 0

            for insts_group in self.policy_to_rollout_recv_insts:
                # insts_group: WeightSyncInstructionsGroup -> inst collection for a full weight tensor
                # handle inst group
                (
                    bytes_received,
                    completion_fn,
                    skipped_cnt,
                ) = self.recv_weight_shard(
                    self.global_rank,
                    insts_group,
                    communicator_index,
                    command.trainable_only,
                    command.do_weight_sync_check,
                )
                skipped_params_cnt += skipped_cnt
                transferred_params_cnt += (
                    len(insts_group.param_instructions) - skipped_cnt
                )
                if (
                    self.weight_mapper.get_unsplited_weight_name(
                        insts_group.param_instructions[0].param_name
                    )
                    != insts_group.param_instructions[0].param_name
                ):
                    # The params in the group of this case originally belong to the same param.
                    # The following counts related with `groups` measure the original params before split.
                    # The count related with `groups` match the count in R2R which is without split.
                    skipped_groups_cnt += 1 if skipped_cnt > 0 else 0
                    transferred_groups_cnt += 0 if skipped_cnt > 0 else 1
                else:
                    skipped_groups_cnt += skipped_cnt
                    transferred_groups_cnt += (
                        len(insts_group.param_instructions) - skipped_cnt
                    )

                pending_bytes[0] += bytes_received
                pending_completions.append(completion_fn)
                total_bytes_received += bytes_received

                pending_groups += 1
                if pending_groups == constant.COSMOS_P2R_NCCL_GROUP_SIZE:
                    nccl_group_end(communicator_index)
                    flush_completions(pending_bytes, pending_completions)
                    nccl_group_start(communicator_index)
                    pending_groups = 0

            nccl_group_end(communicator_index)
            flush_completions(pending_bytes, pending_completions)

            with torch.cuda.stream(copy_stream):
                copy_finished = torch.cuda.Event()
                copy_finished.record()

            copy_finished.wait()

            torch.cuda.synchronize()
            self.total_temp_tensor_pool.clear()

            time_eclapsed = time.time() - st
            logger.info(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received. While {skipped_params_cnt} non-trainable splitted params skipped and {transferred_params_cnt} trainable splitted params transferred."
            )

            if command.trainable_only:
                if not hasattr(self, "p2r_synced_trainable_params_cnt"):
                    self.p2r_synced_trainable_params_cnt = transferred_groups_cnt
                assert (
                    self.p2r_synced_trainable_params_cnt == transferred_groups_cnt
                ), f"Count of trainable unsplitted params which have been synced in P2R {transferred_groups_cnt} must match the synced_trainable_params attribute {self.p2r_synced_trainable_params_cnt}."

            self.state.set_weight_synced()

    @RolloutWorkerBase.register_rollout_command_handler(
        RolloutToRolloutBroadcastCommand
    )
    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names

        # lazy initialization of the vllm engine.
        if self.replica_name != src_replica_name:
            # for replicas that needs to be broadcasted, use dummy format.
            self.lazy_initialize_rollout_engine(load_format="dummy")

        if len(dst_replica_names) > 1:
            self.prepare_trainable_params()
            skipped_params_cnt = 0
            transferred_params_cnt = 0
            logger.info("Starting broadcasting of parameters to all replicas.")
            # Only do broadcast if there are more than one rollout replicas.
            with torch.cuda.stream(self.inference_stream):
                assert (
                    self.rank_in_rollout_repicas >= 0
                ), "[Rollout] rank in rollout replicas should be set before broadcast."
                assert (
                    len(dst_replica_names) == len(self.replica_name_to_rank)
                ), "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."

                src_rank = self.replica_name_to_rank[src_replica_name]
                with torch.inference_mode():
                    for name, parameter in self.rollout.model_param_map(
                        self.weight_mapper
                    ).items():
                        if (
                            name not in self.trainable_params
                            and broadcast_command.trainable_only
                        ):
                            logger.info(
                                f"[Rollout] Skip {name} in R2R due to non trainable."
                            )
                            skipped_params_cnt += 1
                            continue
                        transferred_params_cnt += 1

                        recv_tensor = parameter
                        if not parameter.is_contiguous():
                            recv_tensor = parameter.contiguous()

                        nccl_broadcast(
                            recv_tensor, src_rank, self.global_commnicator_idex
                        )

                        if not parameter.is_contiguous():
                            parameter.copy_(recv_tensor)

                if not self.state.weight_synced():
                    assert not broadcast_command.trainable_only, "[Rollout] Trainable only must be set to False for the first broadcast."
                    self.state.set_weight_synced()

            logger.info(
                f"[Rollout] Finished broadcasting of parameters to all replicas. While {skipped_params_cnt} unsplitted non-trainable params skipped and {transferred_params_cnt} unsplitted params transferred."
            )
            if broadcast_command.trainable_only:
                if not hasattr(self, "r2r_synced_trainable_params_cnt"):
                    self.r2r_synced_trainable_params_cnt = transferred_params_cnt
                if hasattr(self, "p2r_synced_trainable_params_cnt"):
                    # check in R2R sender side.
                    assert (
                        self.r2r_synced_trainable_params_cnt
                        == self.p2r_synced_trainable_params_cnt + len(self.misc_params)
                    ), f"Synced params count in R2R {self.r2r_synced_trainable_params_cnt} must match the sum of count of attribute {self.p2r_synced_trainable_params_cnt} and {len(self.misc_params)}."

        current_step = broadcast_command.weight_step
        if current_step is not None:
            assert (
                current_step >= self.current_weight_version
            ), f"current_step: {current_step} must be greater than or equal to self.current_weight_version: {self.current_weight_version}"
            self.current_weight_version = current_step

        if current_step is not None and current_step > 0:
            should_do_validation = self.config.validation.enable and (
                current_step % self.config.validation.freq == 0
                or current_step == broadcast_command.total_steps
            )

            if should_do_validation:
                self.current_step = current_step
                # Setting the flag, do validation in the main loop.
                self.validation_flag.set()

        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

    def query_command_from_controller(self):
        """Background task to check commands from the controller"""
        while not self.shutdown_signal.is_set():
            commands = []
            try:
                # blocking request
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.error(
                    f"[Rollout] Failed in query commands from controller for replica {self.replica_name}\n: {str(e)}"
                )

            for instruction in commands:
                command = Command.depack(instruction)
                logger.debug(f"[Rollout] Received command: {command.command_type}")
                self._command_queue.put(command)

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            if prompt_queue.empty():
                # blocking request
                payloads, is_end = self.api_client.get_next_prompt(batch_size, **kwargs)
                prompts_and_is_end = (
                    payloads if len(payloads) > 0 else None,
                    is_end,
                )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompts = [
                (prompt[0], RLPayload.model_validate(prompt[1])) for prompt in prompts
            ]
            prompt_queue.put(prompts)
        return is_end

    def consume_one_command(self, cmd_pred: Optional[Callable[[Command], bool]] = None):
        current_command = None
        if self.global_rank == 0:
            if not self._command_queue.empty():
                if cmd_pred is None:
                    current_command = self._command_queue.get()
                else:
                    if cmd_pred(self._command_queue.queue[0]):
                        current_command = self._command_queue.get()
                    else:
                        # Do not go on if the command is not expected
                        current_command = None

        current_command = dist_utils.broadcast_object_cpu(current_command)

        if current_command is not None:
            handler = self.get_rollout_command_handler(type(current_command))
            if handler is None:
                raise Exception(
                    f"No such command supoorted in rollout {current_command}"
                )
            try:
                handler(self, current_command)
                logger.debug(
                    f"[Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Command execution failed for {current_command._serialize()}"
                ) from e
        return current_command

    def consume_command(
        self,
        cmd_pred: Optional[Callable[[Command], bool]] = None,
        timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT,
    ):
        # Consume all pending commands for weight sync.
        # To ensure the weight update is using the up-to-date commands.
        last_cmd = None
        none_cnt = 0
        start_time = time.time()
        while time.time() - start_time < float(timeout):
            cmd = self.consume_one_command(cmd_pred=cmd_pred)
            if cmd is not None:
                last_cmd = cmd
                none_cnt = 0
                start_time = time.time()
            else:
                none_cnt += 1
            if none_cnt >= constant.COSMOS_ROLLOUT_CMD_WAIT_TIMES and (
                (
                    last_cmd is not None
                    and not isinstance(last_cmd, PolicyToRolloutUnicastCommand)
                )
                or last_cmd is None
            ):
                # If continuously get None for COSMOS_ROLLOUT_CMD_WAIT_TIMES times, and the last command is not P2R command, we break.
                # Since P2R must be followed by another R2R broadcast command, we need wait.
                # Continuously get None for COSMOS_ROLLOUT_CMD_WAIT_TIMES times to make sure the command queue is empty at that time.
                break
            time.sleep(constant.COSMOS_ROLLOUT_CMD_WAIT_INTERVAL)

    def send_end_signal(self):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        payloads, is_validation, _, empty = self.report_rollouts(block=True)
        assert (
            not is_validation and payloads is None and empty
        ), f"Payloads must be empty and not for validation when sending end signal {is_validation}, {payloads}, {empty}"
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            prompt_idxs=[],
            payloads=[],
            completions=[],
            is_end=True,
        )
        logger.info(f"[Rollout] Posting rollout end signal to controller: {response}")
        self.api_client.post_rollout_completion(response)

    def report_rollouts(self, block=False):
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    break
                response = RolloutRequest(
                    src_replica_name=self.replica_name,
                    prompt_idxs=[],
                    payloads=payloads,
                    is_end=False,
                )
                self.api_client.post_rollout_completion(response)
            elif not block or empty:
                break
        return payloads, is_validation, step, empty

    @torch.no_grad()
    def main_loop(self):
        while not self.shutdown_signal.is_set():
            self.consume_command(cmd_pred=None)
            if self.validation_flag.is_set():
                # If encounter validation flag during last rollout generation or this command fetch, do validation first.
                self.do_validation()

            # If weight is not ready, nothing else to do.
            if not self.state.weight_synced():
                continue

            # try fetching new prompts if no ending signal is set
            if not self.state.prompt_fetch_end():
                no_more_prompts = self.request_new_prompts(
                    self.batch_size, self._prompt_queue
                )
                if no_more_prompts:
                    logger.info(
                        f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                    )
                    self.state.set_prompt_fetch_end()
                    # Further make sure to set `prompt_consume_end` if no more prompts to be consumed
                    if self._prompt_queue.empty():
                        self.state.set_prompt_consume_end()
                        if self.global_rank == 0:
                            self.send_end_signal()
            _, is_validation, _, _ = self.report_rollouts()
            assert not is_validation, "Validation report should be handled in the broadcast command rather than main loop."
            if self.state.prompt_consume_end():
                assert (
                    self._prompt_queue.empty() and self.state.prompt_fetch_end()
                ), "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                continue
            elif self._prompt_queue.empty():
                continue
            else:
                logger.debug(f"[Rollout] generate start for rank {self.global_rank}")

                # Check if the prompt is valid for the current weight version
                first_payload: RLPayload = self._prompt_queue.queue[0][0][1]
                is_valid_prompt_for_current_weight_version = (
                    first_payload.weight_version <= self.current_weight_version
                )
                if not is_valid_prompt_for_current_weight_version:
                    # Fully Synchronized mode is enabled, we need to wait until the weight version is updated
                    continue

                prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                    self._prompt_queue.get()
                )
                payloads: List[RLPayload] = [
                    payload for _, payload in prompt_id_and_payload_list
                ]

                rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                    payloads=payloads,
                    stream=self.inference_stream,
                    data_packer=self.data_packer,
                    sampling_params=self.sampling_params,
                )

                if len(rollout_results) == 0:
                    continue

                assert len(rollout_results) == len(
                    payloads
                ), f"Error: VLLM returned {len(rollout_results)} for {len(payloads)}"

                # we need filter the result with valid completions or valid completed_conversations
                valid_result: List[RolloutResult] = []
                valid_prompt_id_and_payload_list: List[IdxAndRLPayload] = []
                if self.rollout.rollout_config.multi_turn_config.enable:
                    for id_and_payload, rr in zip(
                        prompt_id_and_payload_list, rollout_results
                    ):
                        valid_conversations: List[ConversationType] = []
                        # remove those result without valid assistant message
                        flag = False
                        for conversation in rr.completed_conversations:
                            for msg in conversation:
                                if msg.role == "assistant" and msg.content != "":
                                    flag = True
                                    break
                            if flag:
                                valid_conversations.append(conversation)
                        rr.completed_conversations = valid_conversations
                        if len(rr.completed_conversations) > 0:
                            valid_result.append(rr)
                            valid_prompt_id_and_payload_list.append(id_and_payload)
                else:
                    # Remove empty completions
                    for id_and_payload, rr in zip(
                        prompt_id_and_payload_list, rollout_results
                    ):
                        completions = rr.completions
                        skip_output = False
                        total_generation_count = len(completions)
                        empty_generation_count = 0
                        output_texts: List[str] = []
                        for j in range(total_generation_count):
                            output_text = completions[j]
                            # if output_text == "":
                            #     logger.warning(
                            #         f"[Rollout] Got empty completion for {i}th prompt {j}th generation"
                            #     )
                            #     empty_generation_count += 1
                            # else:
                            #     output_texts.append(output_text)

                            # Note: (jiaxinc)
                            # We still need to upload the output text, even if it is empty. (replace empty with eos_token)
                            # Because if fully synchronized mode is enabled, we need to make sure the expected
                            # number of global_batch_size is reached at exact time.
                            output_texts.append(
                                output_text
                                if output_text != ""
                                else self.tokenizer.eos_token
                            )
                        # Skip the output if there is one or zero non-empty completions
                        skip_output = (
                            total_generation_count - empty_generation_count
                        ) <= 1
                        if not skip_output:
                            rr.completions = output_texts
                            valid_result.append(rr)
                            valid_prompt_id_and_payload_list.append(id_and_payload)

                logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

                should_report = (
                    self.parallel_dims.tp_coord[0] == 0
                    and (
                        self.parallel_dims.pp_coord[0]
                        == self.parallel_dims.pp_coord[1] - 1
                    )
                    and len(valid_result) > 0
                )

                if should_report:
                    # only the first tp rank in the rollout replica will post the completion to the controller.
                    valid_payloads: List[RLPayload] = []
                    valid_prompt_idxs: List[int] = []

                    for (prompt_idx, old_payload), result in zip(
                        valid_prompt_id_and_payload_list, valid_result
                    ):
                        valid_prompt_idxs.append(prompt_idx)
                        # update payload
                        old_payload.completions = result.completions
                        if self.rollout.rollout_config.multi_turn_config.enable:
                            old_payload.completed_conversations = (
                                result.completed_conversations
                            )
                        valid_payloads.append(old_payload)

                    self.reward_dispatcher.enqueue_rewards_cal(
                        valid_payloads,
                        False,
                        self.current_weight_version,
                        valid_prompt_idxs,
                    )

                if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                    self.state.set_prompt_consume_end()
                    if self.global_rank == 0:
                        self.send_end_signal()
        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def work(self):
        # Start the thread with daemon=True, so it will exit when the main program exits.
        if self.global_rank == 0:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()

        self.main_loop()
        self.inference_stream.synchronize()
        self.handle_shutdown()

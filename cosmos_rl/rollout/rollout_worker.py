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
import requests
import threading
from queue import Queue
import atexit
from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from typing import List, Tuple, Optional, Callable, Any
from functools import partial
from transformers import AutoConfig
from cosmos_rl.rollout import RolloutWorkerBase
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import list_to_b64, b64_to_list
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    Command,
)
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_broadcast,
    nccl_recv,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    WeightSyncInstructionsGroup,
    WeightSyncInstructionsPerParam,
)
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.rollout.backend import determine_backend, RolloutBackend
from cosmos_rl.utils.network_util import make_request_with_retry
import cosmos_rl.utils.util as util
from cosmos_rl.utils import constant
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
    COSMOS_API_VALIDATION_REPORT_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX,
)
import time


"""
Keep in mind that torch distributed is not thread safe. So try to keep the usage in the same thread.
"""


class RolloutWorker(RolloutWorkerBase):
    """
    RolloutWorker will be a replica instance of single DP.
    RolloutWorker should support scaling launch.
    """

    class State:
        UNINITIALIZED = 0
        WEIGHT_SYNCED = 1
        PROMPT_FETCH_END = 1 << 1
        PROMPT_CONSUME_END = 1 << 2

        _state: int = UNINITIALIZED

        def __init__(self):
            self._state = self.UNINITIALIZED

        def weight_synced(self):
            return (self._state & self.WEIGHT_SYNCED) != 0

        def set_weight_synced(self):
            self._state = self._state | self.WEIGHT_SYNCED

        def prompt_fetch_end(self):
            return (self._state & self.PROMPT_FETCH_END) != 0

        def set_prompt_fetch_end(self):
            self._state = self._state | self.PROMPT_FETCH_END

        def prompt_consume_end(self):
            return (self._state & self.PROMPT_CONSUME_END) != 0

        def set_prompt_consume_end(self):
            assert (
                not self.prompt_consume_end()
            ), "Prompt consume end event should not be set twice."
            self._state = self._state | self.PROMPT_CONSUME_END

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super(RolloutWorker, self).__init__(config, parallel_dims)
        self.state = self.State()
        self.config = config
        if self.config.rollout.parallelism.dp_shard_size == -1:
            self.config.rollout.parallelism.dp_shard_size = parallel_dims.dp_shard
        assert self.config.rollout.parallelism.dp_shard_size == parallel_dims.dp_shard
        assert (
            self.config.rollout.parallelism.dp_shard_size > 0
        ), "[Rollout] dp_shard_size should be greater than 0."

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # CommandQueue queried from controller.
        self._command_queue: Queue[Command] = Queue()
        self._prompt_queue: Queue[List[List[int, str]]] = Queue()

        self.rollout_backend = determine_backend(self.config)

        if self.rollout_backend == RolloutBackend.VLLM:
            self.rollout: vLLMRollout = vLLMRollout(
                self.config,
                tokenizer=self.tokenizer,
                seed=self.config.rollout.seed,
                load_format="dummy",
            )
            vLLMRollout.patch_vllm_rollout_locked_step(
                self.rollout,
                self.consume_command,
                self.config.train.enable_validation,
            )

        # communicator index for the cached communicators in C++ binding.
        self.global_commnicator_idex = -1
        # rank in current rollout replicas.
        self.rank_in_rollout_repicas = -1

        # cache for NCCL communicators for P2R.
        self.policy_to_rollout_nccl_communicators = {}

        self.batch_size = self.config.rollout.batch_size
        if self.config.train.enable_validation:
            self.val_batch_size = self.config.rollout.val_batch_size or self.batch_size
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

        if not ModelRegistry.check_model_type_supported(hf_config.model_type):
            raise ValueError(f"Model {hf_config.model_type} not supported.")

        self.weight_mapper = WeightMapper.get_weight_mapper(hf_config.model_type)(
            hf_config
        )
        self.model_config = hf_config

        atexit.register(self.handle_shutdown)

        self.inference_stream = torch.cuda.Stream()

        self.prepare_shard_infos_for_weight_sync_insts()

    def prepare_shard_infos_for_weight_sync_insts(self):
        self.vllm_weight_inplace_view_map, self.recv_param_key_n_rank_list = (
            self.weight_mapper.rollout_prepare_recv(self.get_underlying_model())
        )
        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)
        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        # Ordered list of (hf_key, tensor_dim)
        if self.global_rank == 0:
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "shard_infos": self.all_rank_local_shard_infos,
                        },
                    ),
                    self.get_alternative_urls(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in post shard infos to controller after retries {e}."
                )

    def handle_shutdown(self):
        # Only call once
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True
            if not self.shutdown_signal.is_set():
                self.shutdown_signal.set()
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
            base64_nccl_group_id = list_to_b64(nccl_group_id)
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "unique_pair_name": unique_rollout_group_key,
                            "handle_base64": base64_nccl_group_id,
                        },
                    ),
                    self.get_alternative_urls(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in post nccl group_id to controller after retries {e}."
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
        nccl_group_id = None
        # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
        try:
            r = make_request_with_retry(
                partial(
                    requests.post,
                    json={"unique_pair_name": unique_id_key},
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX),
                max_retries=constant.COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in post nccl group_id to controller after retries {e}."
            )
        base64_nccl_group_id = r.json()["handle_base64"]
        nccl_group_id = b64_to_list(base64_nccl_group_id)
        return nccl_group_id

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_for_param: WeightSyncInstructionsPerParam,
        communicator_index: int,
        do_weight_sync_check: bool = False,
    ):
        total_bytes_received = 0
        dest_name = insts_for_param.param_name
        target_tensor = self.vllm_weight_inplace_view_map[dest_name]
        if do_weight_sync_check:
            cloned_target_tensor = target_tensor.clone()
            # clear the current view
            target_tensor.zero_()

        for inst in insts_for_param.instructions:
            p_rank = inst.policy_rank
            r_rank = inst.rollout_rank
            tensor_split_strategys = inst.slice_strategy
            assert r_rank == global_rank_of_rollout
            view = target_tensor.cosmos_slice(tensor_split_strategys)
            recv_tensor = None
            if view.is_contiguous():
                recv_tensor = view
            else:
                # new a temp tensor
                recv_tensor = torch.empty_like(view)

            nccl_recv(recv_tensor, p_rank, communicator_index)

            # inplace copy
            if not view.is_contiguous():
                view.copy_(recv_tensor)

        if do_weight_sync_check:
            # If the weight sync between Policy and Rollout is correct, the
            # `target_tensor` would have no change.
            # TODO: (lms) When we support quantization in rollout side,
            # we should handle the numerical error of quantized weight, not
            # just apply `torch.allclose` simply.
            if not torch.allclose(cloned_target_tensor, target_tensor):
                raise ValueError(
                    f"Weight sync check failed after weight sync instruction: {insts_for_param} for {dest_name}."
                )

        total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()
        return total_bytes_received

    @RolloutWorkerBase.register_rollout_command_handler(PolicyToRolloutUnicastCommand)
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
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

        if command.do_weight_sync_check:
            self.rollout.reload_weight()

        if not hasattr(self, "policy_to_rollout_recv_insts"):
            self.policy_to_rollout_recv_insts = []
            try:
                insts_meta = make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "rank": self.global_rank,
                        },
                    ),
                    self.get_alternative_urls(
                        COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX
                    ),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
                insts_meta = insts_meta.json()
                self.policy_to_rollout_recv_insts = [
                    WeightSyncInstructionsGroup.from_dict(inst)
                    for inst in insts_meta["insts"]
                ]
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in fetching rollout from policy insts from controller after retries {e}."
                )

        with torch.cuda.stream(self.inference_stream):
            # recv the weight from policy
            st = time.time()
            total_bytes_received = 0
            for insts_group in self.policy_to_rollout_recv_insts:
                for insts_for_per_param in insts_group.param_instructions:
                    total_bytes_received += self.recv_weight_shard(
                        self.global_rank,
                        insts_for_per_param,
                        communicator_index,
                        command.do_weight_sync_check,
                    )
            time_eclapsed = time.time() - st
            logger.debug(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received."
            )
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

        if len(dst_replica_names) > 1:
            # Only do broadcast if there are more than one rollout replicas.
            with torch.cuda.stream(self.inference_stream):
                assert (
                    self.rank_in_rollout_repicas >= 0
                ), "[Rollout] rank in rollout replicas should be set before broadcast."
                assert (
                    len(dst_replica_names) == len(self.replica_name_to_rank)
                ), "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."

                src_rank = self.replica_name_to_rank[src_replica_name]

                for parameter in self.get_underlying_model().parameters():
                    nccl_broadcast(parameter, src_rank, self.global_commnicator_idex)

                if not self.state.weight_synced():
                    self.state.set_weight_synced()

        current_step = broadcast_command.weight_step
        if current_step is not None and current_step > 0:
            should_do_validation = self.config.train.enable_validation and (
                current_step % self.config.train.validation_step == 0
                or current_step == broadcast_command.total_steps
            )
            validation_queue = Queue()
            validation_results = []
            prompt_idxs: List[int] = []
            payloads: List[Any] = []
            if should_do_validation:
                # Do inline validation here
                while True:
                    is_end = self.request_new_prompts(
                        self.val_batch_size,
                        validation_queue,
                        validation_step=current_step,
                    )
                    if not validation_queue.empty():
                        prompts = validation_queue.get()
                        completions: List[List[str]] = self.rollout.rollout_generation(
                            prompt_id_and_payload_list=prompts,
                            stream=self.inference_stream,
                            data_packer=self.val_data_packer,
                        )
                        if completions:
                            prompt_idxs.extend([prompt[0] for prompt in prompts])
                            payloads.extend([prompt[1] for prompt in prompts])
                            validation_results.extend(completions)

                    if is_end:
                        break

                should_report = self.parallel_dims.tp_coord[0] == 0 and (
                    self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
                )

                if should_report:
                    response = ValidationReportRequest(
                        src_replica_name=self.replica_name,
                        validation_step=current_step,
                        prompt_idxs=prompt_idxs,
                        payloads=payloads,
                        completions=validation_results,
                        is_end=True,
                    )
                    try:
                        make_request_with_retry(
                            partial(
                                requests.post,
                                json=response.model_dump(),
                            ),
                            self.get_alternative_urls(
                                COSMOS_API_VALIDATION_REPORT_SUFFIX
                            ),
                            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                        )
                    except Exception as e:
                        logger.error(
                            f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
                        )

        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()

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
            prompt_id_and_payload_list = None
            is_end = False
            url_suffix = COSMOS_API_NEXT_PROMPT_SUFFIX
            try:
                if prompt_queue.empty():
                    # blocking request
                    prompt_meta = make_request_with_retry(
                        partial(
                            requests.get,
                            params={
                                "n": batch_size,
                                **kwargs,
                            },
                        ),
                        self.get_alternative_urls(url_suffix),
                        max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                    )
                    prompt_meta = prompt_meta.json()
                    payload = prompt_meta["prompt_id_and_payload_list"]
                    if len(payload) > 0:
                        prompt_id_and_payload_list = payload
                    is_end = prompt_meta.get("is_end", is_end)
                else:
                    prompt_id_and_payload_list = None
            except Exception as e:
                logger.error(
                    f"[Rollout] Failed in query prompts from controller: {str(e)}"
                )
                prompt_id_and_payload_list = None
            prompts_and_is_end = (prompt_id_and_payload_list, is_end)
            del prompt_id_and_payload_list, is_end

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompt_queue.put(prompts)
        return is_end

    def consume_command(self, cmd_pred: Optional[Callable[[Command], bool]] = None):
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
                raise RuntimeError(f"[Rollout] Command execution failed: {str(e)}")

    def send_end_signal(self, url_suffix: str):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            prompt_idxs=[],
            payloads=[],
            completions=[],
            is_end=True,
        )
        try:
            logger.debug(
                f"[Rollout] Posting rollout end signal to controller: {response}"
            )
            make_request_with_retry(
                partial(
                    requests.post,
                    json=response.model_dump(),
                ),
                self.get_alternative_urls(url_suffix),
                max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
            )

    @torch.no_grad()
    def main_loop(self):
        while not self.shutdown_signal.is_set():
            self.consume_command(cmd_pred=None)

            # If weight is not ready, nothing else to do.
            if not self.state.weight_synced():
                continue

            # try fetching new prompts if no ending signal is set
            if not self.state.prompt_fetch_end():
                no_more_prompts = self.request_new_prompts(
                    self.batch_size, self._prompt_queue
                )
                if no_more_prompts:
                    logger.debug(
                        f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation."
                    )
                    self.state.set_prompt_fetch_end()
                    # Further make sure to set `prompt_consume_end` if no more prompts to be consumed
                    if self._prompt_queue.empty():
                        self.state.set_prompt_consume_end()
                        if self.global_rank == 0:
                            self.send_end_signal(COSMOS_API_ROLLOUT_SUFFIX)

            if self.state.prompt_consume_end():
                assert (
                    self._prompt_queue.empty() and self.state.prompt_fetch_end()
                ), "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                continue
            elif self._prompt_queue.empty():
                continue
            else:
                logger.debug(f"[Rollout] generate start for rank {self.global_rank}")
                prompts: List[Tuple[int, str]] = self._prompt_queue.get()
                completions: List[List[str]] = self.rollout.rollout_generation(
                    prompt_id_and_payload_list=prompts,
                    stream=self.inference_stream,
                    data_packer=self.data_packer,
                )
                logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

                should_report = self.parallel_dims.tp_coord[0] == 0 and (
                    self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
                )

                if should_report and completions is not None:
                    url_suffix = COSMOS_API_ROLLOUT_SUFFIX
                    # only the first tp rank in the rollout replica will post the completion to the controller.
                    prompt_idxs = [prompt[0] for prompt in prompts]
                    payloads = [prompt[1] for prompt in prompts]

                    response = RolloutRequest(
                        src_replica_name=self.replica_name,
                        prompt_idxs=prompt_idxs,
                        payloads=payloads,
                        completions=completions,
                        is_end=False,
                    )
                    try:
                        make_request_with_retry(
                            partial(
                                requests.post,
                                json=response.model_dump(),
                            ),
                            self.get_alternative_urls(url_suffix),
                            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                        )
                    except Exception as e:
                        logger.error(
                            f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
                        )

                if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                    self.state.set_prompt_consume_end()
                    if self.global_rank == 0:
                        self.send_end_signal(COSMOS_API_ROLLOUT_SUFFIX)

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

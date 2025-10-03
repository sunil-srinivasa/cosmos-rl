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

import torch.distributed as dist
import uuid
from typing import Dict, Callable, Type, Optional, Any
import copy
import time
import atexit
import threading
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.network_util import get_local_ip
from cosmos_rl.dispatcher.command import (
    PolicyCommandRegistry,
    RolloutCommandRegistry,
    Command,
)
from cosmos_rl.dispatcher.data.packer import DataPacker, DecoderOnlyLLMDataPacker
from cosmos_rl.dispatcher.data.packer import (
    HFVLMDataPacker,
)

from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.constant as constant
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.dispatcher.protocol import MESH_NAMES
import cosmos_rl.utils.util as util
import base64
import cloudpickle
from transformers import AutoConfig
import multiprocessing as mp
from cosmos_rl.dispatcher.api.client import APIClient


class CommMixin:
    policy_command_handler_registry = PolicyCommandRegistry()
    rollout_command_handler_registry = RolloutCommandRegistry()

    @classmethod
    def register_policy_command_handler(cls, command_type: Type[Command]):
        def decorator(func):
            cls.policy_command_handler_registry.register(command_type, func)
            return func

        return decorator

    @classmethod
    def register_rollout_command_handler(
        cls, command_type: Type[Command], backend: str = "vllm"
    ):
        def decorator(func):
            cls.rollout_command_handler_registry.register(command_type, func, backend)
            return func

        return decorator

    @classmethod
    def get_policy_command_handler(
        cls, command_type: Type[Command]
    ) -> Optional[Callable]:
        return cls.policy_command_handler_registry.get_command_handler(command_type)

    @classmethod
    def get_rollout_command_handler(
        cls, command_type: Type[Command], backend: str = "vllm"
    ) -> Optional[Callable]:
        return cls.rollout_command_handler_registry.get_command_handler(
            command_type, backend
        )

    def init_comm(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        logger.info(
            f"{self.role} Replica started at global rank {self.global_rank}, with replica name: {self.replica_name}"
        )

        self.api_client = APIClient(self.role)
        self.init_meta()

        self.register_to_controller()

    def init_meta(self):
        # Fetch metadata from the controller
        metadata = self.api_client.get_controller_metadata()
        self.init_data_packer(metadata)

    def init_data_packer(self, metadata: Dict[str, Any]):
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        model_type = hf_config.model_type

        user_data_packer = metadata.get("user_data_packer", None)
        if user_data_packer:
            user_data_packer = base64.b64decode(user_data_packer)
            user_data_packer = cloudpickle.loads(user_data_packer)
            self.data_packer = user_data_packer
            logger.info(f"Using user-provided data packer: {self.data_packer}")
        else:
            try:
                self.data_packer = DataPacker.get_default_data_packer(model_type)
                logger.info(f"Using default data packer: {self.data_packer}")
            except ValueError:
                self.data_packer = (
                    DecoderOnlyLLMDataPacker() if not is_vlm else HFVLMDataPacker()
                )
                logger.warning(
                    f"No default data packer found for {model_type}, using {type(self.data_packer).__name__} as default"
                )

        self.data_packer.setup(self.config, self.tokenizer)

        user_val_data_packer = metadata.get("user_val_data_packer", None)
        if user_val_data_packer:
            user_val_data_packer = base64.b64decode(user_val_data_packer)
            user_val_data_packer = cloudpickle.loads(user_val_data_packer)
            self.val_data_packer = user_val_data_packer
            logger.info(
                f"Using user-provided validation data packer: {self.val_data_packer}"
            )
        else:
            try:
                self.val_data_packer = DataPacker.get_default_data_packer(model_type)
                logger.info(
                    f"Using default validation data packer: {self.val_data_packer}"
                )
            except ValueError:
                self.val_data_packer = (
                    DecoderOnlyLLMDataPacker() if not is_vlm else HFVLMDataPacker()
                )
                logger.warning(
                    f"No default validation data packer found for {model_type}, using {type(self.val_data_packer).__name__} as default"
                )

        self.val_data_packer.setup(self.config, self.tokenizer)

    def register_to_controller(self):
        if hasattr(self, "_is_registered"):
            return

        target_mesh_names = copy.deepcopy(MESH_NAMES)
        ranks = []
        group_size = []
        for mesh_name in MESH_NAMES:
            if (
                self.parallel_dims.mesh.mesh_dim_names
                and mesh_name in self.parallel_dims.mesh.mesh_dim_names
            ):
                ranks.append(self.parallel_dims.mesh[mesh_name].get_local_rank())
                group_size.append(self.parallel_dims.mesh[mesh_name].size())
            else:
                ranks.append(0)
                group_size.append(1)

        host_info_tuple = get_local_ip()
        if host_info_tuple is None:
            raise Exception("Failed to get local IP address")
        host_ip, host_name = host_info_tuple
        self.api_client.register(
            replica_name=self.replica_name,
            role=self.role,
            mesh_names=target_mesh_names,
            ranks=ranks,
            group_size=group_size,
            global_rank=self.global_rank,
            host_ip=host_ip,
            host_name=host_name,
        )

        dist.barrier()  # wait all the atoms registered.

        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event

        if self.global_rank == 0:
            logger.info(
                f"{self.role} Replica {self.replica_name} registered to controller"
            )
            # Start the thread with daemon=True, so it will exit when the main program exits.
            process = mp.Process(
                target=self.heartbeat_trigger,
                args=(self.shutdown_mp_signal,),
                daemon=True,  # Dies when main process exits
            )
            process.start()
            self.heartbeat_thread = process
        else:
            self.heartbeat_thread = None

        self._is_registered = True
        atexit.register(self.unregister_from_controller)

    def unregister_from_controller(self):
        if not hasattr(self, "_is_registered"):
            return
        elif hasattr(self, "_is_unregistered"):
            return
        else:
            self._is_unregistered = True
        self._is_registered = False
        # let only rank == 0 send the unregister request
        if self.global_rank == 0:
            self.api_client.unregister(self.replica_name)

    def get_group_unique_key(self, replica_name_to_rank: Dict[str, int]):
        return (
            "_".join(
                [
                    k
                    for k, _ in sorted(
                        replica_name_to_rank.items(), key=lambda item: item[1]
                    )
                ]
            )
            + "_"
            + str(self.global_rank)
        )

    def init_redis(self):
        assert (
            self.api_client.remote_ips is not None
        ), "Please init the api client first"
        # For command fetch via redis connection
        self.redis_controller = RedisStreamHandler(
            ips=self.api_client.remote_ips, port=int(self.config.redis)
        )
        logger.debug(
            f"[{self.role}] Init redis at {self.api_client.remote_ips}:{self.redis_controller.port}"
        )

    def heartbeat_trigger(self, shutdown_signal: threading.Event):
        while True:
            self.api_client.post_heartbeat(self.replica_name)

            # If the heartbeat interval is greater than 1, we need to check the shutdown signal every second
            # for faster shutdown check
            if constant.COSMOS_HEARTBEAT_SEND_INTERVAL > 1:
                early_break = False
                for _ in range(int(constant.COSMOS_HEARTBEAT_SEND_INTERVAL)):
                    if shutdown_signal.is_set():
                        early_break = True
                        break
                    else:
                        time.sleep(1)
                if early_break:
                    break
            else:
                time.sleep(constant.COSMOS_HEARTBEAT_SEND_INTERVAL)
                if shutdown_signal.is_set():
                    break

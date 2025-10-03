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

import sys
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as RolloutConfig
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_worker import vLLMRolloutWorker
from cosmos_rl.dispatcher.api.client import APIClient


def run_rollout(*args, **kwargs):
    api_client = APIClient(role="ROLLOUT")
    metadata = api_client.get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Rollout] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
        )

    cosmos_rollout_config = RolloutConfig.from_dict(
        metadata["config"]
    )  # just use config as key temporarily

    task_type = cosmos_rollout_config.train.train_policy.type
    if task_type not in ["grpo"]:
        logger.info(
            "[Rollout] Task in controller is not type of Reinforcement Learning. Aborted."
        )
        sys.exit(0)

    logger.info(
        f"[Rollout] Loaded rollout configuration: {cosmos_rollout_config.rollout.model_dump()}"
    )

    try:
        rollout_backend = cosmos_rollout_config.rollout.backend
        if rollout_backend == "vllm":
            parallel_dims = ParallelDims.from_config(
                parallesim_config=cosmos_rollout_config.rollout.parallelism
            )
            init_distributed()
            parallel_dims.build_mesh(device_type="cuda")
            rollout_worker = vLLMRolloutWorker(cosmos_rollout_config, parallel_dims)
            rollout_worker.setup(
                dataset=kwargs.get("dataset"),
                reward_fns=kwargs.get("reward_fns"),
                filter_reward_fns=kwargs.get("filter_reward_fns"),
                val_dataset=kwargs.get("val_dataset"),
                val_reward_fns=kwargs.get("val_reward_fns"),
            )
        elif rollout_backend == "trtllm":
            try:
                from cosmos_rl.rollout.trtllm_rollout.trtllm_rollout_wrapper import (
                    TRTLLMRolloutWrapper,
                )
            except ImportError as e:
                logger.error(f"[Rollout] TRTLLMRolloutWrapper importing failed! {e}")
                raise e
            # if backend is trtllm, we leave distribution initialization to trtllm executor.
            rollout_worker = TRTLLMRolloutWrapper(cosmos_rollout_config)
            rollout_worker.setup(
                dataset=kwargs.get("dataset"),
                reward_fns=kwargs.get("reward_fns"),
                filter_reward_fns=kwargs.get("filter_reward_fns"),
                val_dataset=kwargs.get("val_dataset"),
                val_reward_fns=kwargs.get("val_reward_fns"),
            )
        else:
            raise ValueError(f"Invalid rollout backend: {rollout_backend}")
        rollout_worker.work()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        del rollout_worker
        destroy_distributed()
        logger.info("[Rollout] Destroy context of torch dist.")


if __name__ == "__main__":
    run_rollout()

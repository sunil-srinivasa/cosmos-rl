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

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.policy.trainer.sft_trainer import SFTTrainer
from cosmos_rl.policy.trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.config import Config as CosmosConfig
import torch
from cosmos_rl.utils import util
from cosmos_rl.dispatcher.api.client import APIClient


def main(*args, **kwargs):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    api_client = APIClient(role="POLICY")
    metadata = api_client.get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Policy] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
        )

    cosmos_config = CosmosConfig.from_dict(metadata["config"])

    logger.info(f"[Policy] Loaded configuration: {cosmos_config.model_dump()}")

    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    policy_type = cosmos_config.train.train_policy.type

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=util.str2torch_dtype(cosmos_config.train.param_dtype),
        ):
            if policy_type == "grpo":
                logger.info("Starting GRPO training...")
                trainer = GRPOTrainer(config=cosmos_config, parallel_dims=parallel_dims)
                trainer.main_loop()
            elif policy_type == "sft":
                custom_sft_dataset = kwargs.get("dataset")
                custom_sft_data_packer = kwargs.get("data_packer")
                logger.info("Starting SFT training...")
                trainer = SFTTrainer(
                    config=cosmos_config,
                    parallel_dims=parallel_dims,
                    dataset=custom_sft_dataset,
                    data_packer=custom_sft_data_packer,
                    val_dataset=kwargs.get("val_dataset", None),
                    val_data_packer=kwargs.get("val_data_packer", None),
                    sampler=kwargs.get("sampler", None),
                    batch_sampler=kwargs.get("batch_sampler", None),
                    val_sampler=kwargs.get("val_sampler", None),
                    val_batch_sampler=kwargs.get("val_batch_sampler", None),
                )
                trainer.train()
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
    finally:
        destroy_distributed()
        logger.info("Process group destroyed.")


if __name__ == "__main__":
    main()

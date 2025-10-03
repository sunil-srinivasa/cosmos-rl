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

from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import ParallelismConfig
import contextlib
from typing import Generator, Optional, List
import torch
import math
import numpy
import os


def train_context(enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import sdpa_kernel, SDPBackend

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def unshard_context_parallel_output(
    cp_mesh: DeviceMesh, buffers: List[torch.Tensor], cp_seq_dims: List[int]
) -> torch.Tensor:
    try:
        from torch.distributed.tensor.experimental._attention import (
            context_parallel_unshard,
        )
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    return context_parallel_unshard(cp_mesh, buffers, cp_seq_dims)


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    pp_dynamic_shape: bool
    ep: int = 1
    # When ep is enabled, we can have different dp shard for the MoE module.
    # For example, suppose we have 64 GPUs, then we can have dp_shard equal
    # to 64 for the attention module, and have ep = 4, dp_shard_with_ep = 16
    # for the MoE module.
    dp_shard_with_ep: int = -1

    @staticmethod
    def from_config(parallesim_config: ParallelismConfig):
        return ParallelDims(
            dp_replicate=parallesim_config.dp_replicate_size,
            dp_shard=parallesim_config.dp_shard_size,
            cp=parallesim_config.cp_size,
            tp=parallesim_config.tp_size,
            pp=parallesim_config.pp_size,
            ep=parallesim_config.ep_size,
            world_size=parallesim_config.world_size,
            pp_dynamic_shape=parallesim_config.pp_dynamic_shape,
        )

    @staticmethod
    def from_config_for_analysis(parallesim_config: ParallelismConfig, world_size: int):
        return ParallelDims(
            dp_replicate=parallesim_config.dp_replicate_size,
            dp_shard=parallesim_config.dp_shard_size,
            cp=parallesim_config.cp_size,
            tp=parallesim_config.tp_size,
            pp=parallesim_config.pp_size,
            ep=parallesim_config.ep_size,
            world_size=world_size,
            pp_dynamic_shape=parallesim_config.pp_dynamic_shape,
        )

    def __post_init__(self):
        self._validate()
        self.build_mesh_info()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep, dp_shard_with_ep, world_size = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.dp_shard_with_ep,
            self.world_size,
        )
        for d in (dp_replicate, cp, tp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must be -1 or >=1."
        if dp_shard < 0:
            logger.info(
                "dp_shard is set to -1, will be automatically determined based on "
                f"world_size {world_size} // {pp * dp_replicate * cp * tp}."
            )

            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert (
            dp_shard >= 1
        ), f"dp_shard of size {dp_shard} is not valid, should be equal or greater than 1"

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if pp > 1 and dp_replicate > 1:
            raise ValueError(
                "dp_replicate must be 1 when pp > 1, since we only support FSDP with pipeline parallelism."
            )

        # Checks for MoE weights. Note that dp_shard is only used for the non-MoE weights.
        if ep > 1:
            assert (
                dp_shard_with_ep == -1 or dp_shard_with_ep >= 1
            ), " dp_shard_with_ep must -1 or >=1."
            if dp_shard_with_ep < 0:
                logger.info(
                    "dp_shard_with_ep is set to -1, will be automatically determined based "
                    f"on self.world_size {self.world_size} // {pp * ep * tp}."
                )
                self.dp_shard_with_ep = dp_shard_with_ep = self.world_size // (
                    pp * ep * tp
                )
                logger.info(f"dp_shard_with_ep is set to {dp_shard_with_ep}.")
            assert (
                dp_shard_with_ep >= 1
            ), f"WORLD_SIZE({self.world_size}) is not a multiple of pp({pp}) * ep({ep}) * tp({tp})"

            if tp > 1:
                raise ValueError(
                    "tp must be 1 when ep > 1, since we only support MoE with pipeline parallelism."
                )

            if pp > 1 and dp_shard_with_ep > 1:
                raise ValueError(
                    "dp_shard_with_ep must be 1 when pp > 1, since we only support EP with pipeline parallelism."
                )

            if (pp * dp_shard_with_ep * ep * tp) != self.world_size:
                raise ValueError(
                    f"Invalid parallel dims: pp({pp}) * dp_shard_with_ep({dp_shard_with_ep}) * "
                    f"ep({ep}) * tp({tp}) != WORLD_SIZE({self.world_size})"
                )

    def _build_mesh(
        self, device_type: str, dims: list[int], names: list[str]
    ) -> DeviceMesh:
        valid_dims = []
        valid_names = []
        for dim, name in zip(dims, names):
            if dim > 1:
                valid_dims.append(dim)
                valid_names.append(name)

        assert (
            math.prod(valid_dims) == self.world_size
        ), f"Invalid parallel dims: prod({valid_dims}) != WORLD_SIZE({self.world_size})"

        logger.info(
            f"Building {len(valid_dims)}-D device mesh with {valid_names}, {valid_dims}"
        )
        return init_device_mesh(device_type, valid_dims, mesh_dim_names=valid_names)

    def build_mesh(self, device_type: str) -> DeviceMesh:
        mesh = self._build_mesh(
            device_type,
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            [
                "pp",
                "dp_replicate",
                "dp_shard",
                "cp",
                "tp",
            ],  # reverse order to apply N-dim parallel.
        )

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh useful for TP-merged FSDP
        dp_cp_tp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
            # Do not add dp_replicate to dp_cp_tp_mesh_dim_names,
            # since each replica is independent of each other
            # dp_cp_tp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_tp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_tp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")
        if self.tp_enabled:
            dp_cp_tp_mesh_dim_names.append("tp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_tp_mesh_dim_names != []:
            mesh[tuple(dp_cp_tp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp_tp")
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        self.mesh = mesh
        return mesh

    def build_meshes_with_ep(self, device_type: str) -> dict[str, DeviceMesh]:
        meshes = {}
        meshes["default"] = self.build_mesh(device_type)

        if self.ep > 1:
            meshes["moe"] = self._build_mesh(
                device_type,
                [self.pp, self.dp_shard_with_ep, self.ep],
                ["pp", "dp_shard_with_ep", "ep"],
            )

        return meshes

    def get_rank_in_dim(self, mesh_dim_name: str, global_rank: int) -> int:
        if hasattr(self, "full_rank_info"):
            if mesh_dim_name in self.full_rank_info[global_rank]:
                return self.full_rank_info[global_rank][mesh_dim_name]
            else:
                raise ValueError(f"Mesh dim {mesh_dim_name} not found in rank info.")
        else:
            raise ValueError(
                "full_rank_info is not set. Please call build_mesh() first."
            )

    def get_size_in_dim(self, mesh_dim_name: str) -> int:
        if hasattr(self, "full_world_size_info"):
            if mesh_dim_name in self.full_world_size_info:
                return self.full_world_size_info[mesh_dim_name]
            else:
                raise ValueError(
                    f"Mesh dim {mesh_dim_name} not found in world size info."
                )
        else:
            try:
                return self.mesh.get_group(mesh_dim_name).size()
            except Exception:
                pass
            return 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def dp_shard_with_ep_enabled(self) -> bool:
        return self.dp_shard_with_ep > 1

    @property
    def pp_dynamic_shape_enabled(self) -> bool:
        return self.pp > 1 and self.pp_dynamic_shape

    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def dp_replicate_coord(self):
        if not self.dp_replicate_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="dp_replicate"), self.dp_replicate)

    @property
    def tp_coord(self):
        if not self.tp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="tp"), self.tp)

    @property
    def pp_coord(self):
        if not self.pp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="pp"), self.pp)

    @property
    def dp_shard_coord(self):
        if not self.dp_shard_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="dp_shard"), self.dp_shard)

    @property
    def cp_coord(self):
        if not self.cp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="cp"), self.cp)

    @property
    def dp_shard_cp_coord(self):
        if not self.dp_shard_enabled and not self.cp_enabled:
            return 0, 1
        else:
            return self.mesh[tuple(("dp_shard_cp",))].get_local_rank(), self.mesh[
                tuple(("dp_shard_cp",))
            ].size()

    def build_mesh_info(self):
        dims = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]
        dim_paras = {
            "pp": self.pp,
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "cp": self.cp,
            "tp": self.tp,
        }
        info = [{} for i in range(self.world_size)]
        meshes = [range(self.world_size)]
        for dim in dims:
            new_meshes = []
            for m in meshes:
                for r, arr in enumerate(numpy.array_split(m, dim_paras[dim])):
                    for d in list(arr):
                        if d in m:
                            info[d][dim] = r
                    new_meshes.append(list(arr))
            meshes = new_meshes
        # Note: full_rank_info will record the rank in each dimension for a global rank/device.
        # e.g: [{'pp': 0, 'dp_replicate': 0, 'dp_shard': 0, 'cp': 0, 'tp': 0, 'dp_shard_cp': 0, 'dp': 0},
        # {'pp': 0, 'dp_replicate': 0, 'dp_shard': 0, 'cp': 0, 'tp': 1, 'dp_shard_cp': 0, 'dp': 0}]
        self.full_rank_info = info
        self.full_world_size_info = dim_paras
        self.full_world_size_info["dp_shard_cp"] = self.dp_shard * self.cp
        self.full_world_size_info["dp"] = self.dp_replicate * self.dp_shard
        self.full_world_size_info["dp_cp_tp"] = (
            self.dp_replicate * self.dp_shard * self.cp * self.tp
        )

        for i in range(self.world_size):
            self.full_rank_info[i]["dp_cp_tp"] = (
                self.full_rank_info[i]["dp_replicate"]
                * self.dp_shard
                * self.cp
                * self.tp
                + self.full_rank_info[i]["dp_shard"] * self.cp * self.tp
                + self.full_rank_info[i]["cp"] * self.tp
                + self.full_rank_info[i]["tp"]
            )
            self.full_rank_info[i]["dp_shard_cp"] = (
                self.full_rank_info[i]["dp_shard"] * self.cp
                + self.full_rank_info[i]["cp"]
            )
            self.full_rank_info[i]["dp"] = (
                self.full_rank_info[i]["dp_replicate"] * self.dp_shard
                + self.full_rank_info[i]["dp_shard"]
            )

        self.global_rank = int(os.environ.get("RANK", 0))
        # logger.info(f"Full rank info: {self.full_rank_info}")

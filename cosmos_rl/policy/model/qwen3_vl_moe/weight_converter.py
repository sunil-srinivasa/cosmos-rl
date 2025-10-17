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

import re
from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple


def map_key_from_hf(name: str, src_model_type: str) -> str:
    if src_model_type in ["qwen3_vl_moe"]:
        prefix = None
        if name.startswith("model.language_model."):
            prefix = "model.language_model."
        elif name.startswith("model.visual."):
            prefix = "model.visual."
        elif name.startswith("lm_head."):
            return name
        else:
            raise ValueError(f"Unsupported weight: {name}")
        return name.replace(prefix, "")
    else:
        raise ValueError(f"Unsupported model type: {src_model_type}")


def qwen3_moe_lm_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    n_experts: int,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    tp_ep_rank, tp_ep_size = parallel_dims.tp_coord
    assert n_experts % tp_ep_size == 0, "n_experts must be divisible by tp_ep_size"

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    # Expert weight are aggregated into (n_experts, in_features, out_features)
    # Weight are loaded in (out_features, in_features) shape
    # So we do not do FSDP sharding on expert weights, instead we filter by expert id
    should_do_fsdp_sharding = True

    dest_name = map_key_from_hf(name, src_model_type)

    if "lm_head.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif "lm_head.bias" == dest_name:
        shard = tensor
    elif "embed_tokens.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif dest_name in ["norm.weight", "norm.bias"]:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(r"layers\.(\d+)\.input_layernorm\.(weight|bias)", dest_name)
    ) is not None:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(o_proj)\.(weight|bias)", dest_name
        )
    ) is not None:
        if dest_name.endswith(".bias"):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)",
            dest_name,
        )
    ) is not None:
        should_do_fsdp_sharding = False
        shard = tensor
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", dest_name
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.gate\.weight", dest_name
        )
    ) is not None:
        # TODO(cjx): Small enough, forbid FSDP sharding is better
        shard = tensor
    elif not ignore_unknown_weights:
        raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        return None, None

    # Do FSDP sharding
    shard = shard.contiguous()
    if should_do_fsdp_sharding:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    lm_type: str,
    n_experts: int,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    # For LM
    lm_part_name, lm_part_shard = qwen3_moe_lm_weight_from_hf(
        tensor,
        name,
        src_model_type,
        n_experts,
        parallel_dims,
        ignore_unknown_weights=True,
    )
    if lm_part_name is not None:
        return lm_part_name, lm_part_shard

    # For Visual
    visual_prefix = "model.visual."
    assert name.startswith(visual_prefix), f"Unsupported weight: {name}"

    if (
        parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
    ):
        dp_shard_rank = parallel_dims.mesh["dp_cp_tp"].get_local_rank()
        dp_shard_size = parallel_dims.mesh["dp_cp_tp"].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = name.replace(visual_prefix, "")
    shard = tensor
    # Do FSDP sharding only for visual part
    shard = shard.contiguous()
    if shard.shape[0] % dp_shard_size == 0:
        shard = shard.tensor_split(dp_shard_size, dim=0)
        shard = shard[dp_shard_rank]
    else:
        chunk_size = (shard.shape[0] + dp_shard_size - 1) // dp_shard_size
        shard = shard[dp_shard_rank * chunk_size : (dp_shard_rank + 1) * chunk_size]

    return dest_name, shard.contiguous()

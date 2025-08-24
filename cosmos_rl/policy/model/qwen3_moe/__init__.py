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
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from transformers import AutoConfig
import torch.distributed._symmetric_memory as symm_mem
from cosmos_rl.utils.util import (
    resolve_model_path,
    IdentityLayer,
    clear_weight_name,
    sync_model_vocab,
    retry,
)
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.qwen3_moe.weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.model.qwen3_moe.weight_mapper import Qwen3MoeWeightMapper
from cosmos_rl.policy.kernel.symm_mem_recipes import OnDeviceAllToAllV
from cosmos_rl.policy.kernel.moe.indices import generate_permute_indices
from cosmos_rl.policy.kernel.moe.grouped_gemm import group_gemm_imp
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from functools import cached_property
import cosmos_rl.policy.kernel.modeling_utils as modeling_utils
from cosmos_rl.policy.kernel.norm import RMSNorm
import cosmos_rl.policy.kernel.rope as rope
from cosmos_rl.utils.sequence_packing import pack_sequences_for_inputs


def build_norm(
    norm_type: str, dim: int, eps: float, casting_mode: Optional[str] = None
):
    assert norm_type == "rmsnorm", f"Unknown norm_type: '{norm_type}'"
    return RMSNorm(dim, eps, casting_mode=casting_mode)


@dataclass
class Qwen3MoeArgs:
    dim: int
    ffn_dim: int
    n_layers: int
    n_experts: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    biases: List[str] = field(default_factory=lambda: [])
    q_k_norm_enabled: bool = False
    norm_eps: float = 1e-6
    rope_theta: float = 10000
    norm_type: str = "rmsnorm"
    rope_type: str = "default"
    hf_config: AutoConfig = None


class RotaryEmbedding(nn.Module):
    def __init__(self, args: Qwen3MoeArgs, device=None):
        super().__init__()
        self.args = args
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[args.rope_type]
        self.device = device
        self.config = args
        self.reset_inv_freq(device=device)

    def reset_inv_freq(self, device: torch.device = None):
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config.hf_config, self.device
        )
        if not hasattr(self, "inv_freq"):
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq.to(torch.float32)
            with torch.no_grad():
                self.inv_freq.data.copy_(inv_freq)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq.dtype != torch.float32:
            self.reset_inv_freq(device=x.device)
            assert self.inv_freq.dtype == torch.float32

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (Qwen3MoeArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        q_proj (Linear): Linear transformation for queries.
        k_proj (Linear): Linear transformation for keys.
        v_proj (Linear): Linear transformation for values.
        o_proj (Linear): Linear transformation for output.
    """

    def __init__(self, model_args: Qwen3MoeArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

        self.q_proj = nn.Linear(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias="q_proj" in model_args.biases,
        )
        self.q_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )

        self.k_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="k_proj" in model_args.biases,
        )
        self.k_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )

        self.v_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="v_proj" in model_args.biases,
        )
        self.o_proj = nn.Linear(
            model_args.n_heads * self.head_dim,
            model_args.dim,
            bias="o_proj" in model_args.biases,
        )
        self.rope_func = rope.RotaryPositionEmbedding()

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths.
            max_seqlen (int, optional): Maximum sequence length.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.q_norm is not None:
            xq = self.q_norm(xq.view(bs, seqlen, -1, self.head_dim))
        if self.k_norm is not None:
            xk = self.k_norm(xk.view(bs, seqlen, -1, self.head_dim))

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = self.rope_func(xq, xk, cos, sin, unsqueeze_dim=2)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                raise ValueError("Flash attention only supports float32 input")
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        if cu_seqlens is not None:
            xq = xq.view(seqlen, -1, self.head_dim)
            xk = xk.view(seqlen, -1, self.head_dim)
            xv = xv.view(seqlen, -1, self.head_dim)
            output = modeling_utils.flash_attn_varlen_func(
                xq,
                xk,
                xv,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=True,
            )
        else:
            output = modeling_utils.flash_attn_func(xq, xk, xv, causal=True)
        output = output.view(bs, seqlen, -1)
        return self.o_proj(output)


class MoEGate(nn.Module):
    def __init__(
        self,
        num_routed_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        dim: int,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob
        self.num_routed_experts = num_routed_experts
        self.weight = nn.Parameter(torch.empty((self.num_routed_experts, dim)))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight


class FakeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))


class FeedForward(nn.Module):
    """
    FeedForward module, support hybrid parallelism including:
    - TP: Shard the experts row/col-wisely across TP groups
    - EP: split the experts into groups, located on EP groups
    - FSDP: Shard the weights across FSDP groups

    Args:
        dim (int): Input dimension.
        intermediate_dim (int): Intermediate dimension.
        model_args (Qwen3MoeArgs): Model configuration arguments.
    """

    token_send_buf: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_id: int,
        model_args: Qwen3MoeArgs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.total_experts = model_args.n_experts
        self.local_experts = model_args.n_experts
        self.intermediate_dim = intermediate_dim
        self.dim = dim
        self.up_proj = FakeLinear(dim, intermediate_dim, self.local_experts)
        self.gate_proj = FakeLinear(dim, intermediate_dim, self.local_experts)
        self.down_proj = FakeLinear(intermediate_dim, dim, self.local_experts)
        self.act_fn = F.silu
        self.gate = MoEGate(
            num_routed_experts=self.total_experts,
            num_experts_per_tok=model_args.hf_config.num_experts_per_tok,
            norm_topk_prob=model_args.hf_config.norm_topk_prob,
            dim=dim,
        )
        self.local_to_dtensor = IdentityLayer()
        self.reshard_helper_layer = IdentityLayer()
        self.group_gemm_imp = group_gemm_imp()

        assert not any(
            [
                "up_proj" in model_args.biases,
                "gate_proj" in model_args.biases,
                "down_proj" in model_args.biases,
            ]
        ), "up_proj, gate_proj, and down_proj cannot be in biases for Qwen3Moe"

    def sort_tokens(self, x, topk_ids, topk_weights):
        # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
        # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.
        # Since this is an "aritificial" index creation (final outcome being
        # `idxs`), we don't need gradients here.

        with torch.no_grad():
            # [seq_len, n_routed_experts]
            expert_counts = topk_ids.new_zeros((topk_ids.shape[0], self.total_experts))
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort()

        sorted_tokens = x[token_indices // topk_ids.shape[1]]
        # assert sorted_tokens.shape == sorted_tokens_shape

        return (sorted_tokens, token_indices, tokens_per_expert)

    def get_send_buf(self):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_gather_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def moe_on_device(self, x, topk_ids, topk_weight):
        """
        x: [batch * local_seq_len, dim]
        topk_ids: [batch * local_seq_len, topk]
        topk_weight: [batch * local_seq_len, topk]

        sorted_tokens: [batch * local_seq_len * topk, dim]
        token_indices: [batch * local_seq_len * topk]
        tokens_per_expert: [n_experts]
        """
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)
        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            # tokens_per_expert: [n_experts, 1]
            # tokens_per_expert_group: [n_experts, 1]
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            # For TP/EP mode, the input is sequencely parallelized
            # So each EP group will have distinct, but the same number of tokens
            # After this collective, tokens_per_expert_group is still of shape [n_experts, 1]

            # Let's say we are on EP rank 0:
            # recv: [(e0, e1, e2 ...), (e0, e1, e2 ...), ...], totally `n_experts` elements
            #        ----------------: tokens from EP group 0 to EP group 0
            #                          ----------------: tokens from EP group 1 to EP group 0
            #                          ...
            # So we can just concat
            dist.all_to_all_single(
                tokens_per_expert_group,
                tokens_per_expert,
                group=self.ep_group,
                async_op=False,
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        # Move input to the `token_send_buf` symm mem
        token_send_buf = self.get_send_buf()
        token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
        # Note: `out=` avoids copy, but it is not differentiable
        # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=token_send_buf[: idxs.shape[0]])

        # Reference:
        #   1. [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/deepseek_v3/symm_mem_recipes/triton_on_device_all_to_all_v.py)
        #   2. [Symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes)
        token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
            token_send_buf,
            input_splits,
            self.ep_group,
        )

        # We need to permute the received tokens so that tokens for the same expert are contiguous.
        # This part prepares a 1D tensor `permuted_indices` for such permutation.
        # This part doesn't need gradient.
        with torch.no_grad():
            ALIGN_SIZE_M = 128
            permuted_indices, m_sizes, m_offsets = generate_permute_indices(
                tokens_per_expert_group,
                self.local_experts,
                self.ep_size,
                ALIGN_SIZE_M,
            )
        # Permute the received tokens so that tokens for the same expert are contiguous.
        contig_tokens = token_gather_buf[permuted_indices]
        # group gemm - handle all three group gemms (up, gate, down for all experts)
        # print(f"m_sizes: {m_sizes}, m_offsets: {m_offsets}")
        hidden_outputs = self.group_gemm_imp(
            contig_tokens,
            m_sizes,
            m_offsets,
            self.gate_proj.weight.to_local(),
            self.up_proj.weight.to_local(),
            self.down_proj.weight.to_local(),
            self.act_fn,
        )

        # Prepare buffer for tokens processed by experts
        processed_tokens = self.get_gather_buf()

        # Move into Symmetric Memory for the return shuffle
        processed_tokens[permuted_indices] = hidden_outputs

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        token_return_buf, _ = OnDeviceAllToAllV.apply(
            processed_tokens,
            output_splits,
            self.ep_group,
        )

        returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens

        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )

        return final_out

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [bsz, seqlen // ep_size, dim]
        """
        assert self.ep_group is not None, "EP group is not set"
        orig_shape = hidden_states.shape
        # topk_idx: [batch * local_seq_len, topk]
        # topk_weight: [batch * local_seq_len, topk]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        y = self.moe_on_device(hidden_states, topk_idx, topk_weight)
        y = y.view(*orig_shape)
        return self.reshard_helper_layer(y)


class Qwen3MoEBlock(nn.Module):
    """
    Qwen3MoEBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (Qwen3MoeArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: Qwen3MoeArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.self_attn = Attention(model_args)
        self.mlp = FeedForward(
            dim=model_args.dim,
            intermediate_dim=model_args.ffn_dim,
            model_args=model_args,
            layer_id=layer_id,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.input_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )
        self.post_attention_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ):
        """
        Perform a forward pass through the Qwen3MoEBlock.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.self_attn(
            self.input_layernorm(x),
            position_embeddings,
            cu_seqlens=kwargs.get("cu_seqlens", None),
            max_seqlen=kwargs.get("max_seqlen", None),
        )
        out = self.mlp(self.post_attention_layernorm(h))
        out = h + out
        return out


@ModelRegistry.register(Qwen3MoeWeightMapper)
class Qwen3MoE(BaseModel):
    """
    Qwen3MoE Module

    Args:
        model_args (Qwen3MoeArgs): Model configuration arguments.

    Attributes:
        model_args (Qwen3MoeArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        embed_tokens (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Qwen3Moe blocks.
        norm (RMSNorm): Layer normalization for the model output.
        lm_head (ColumnParallelLinear): Linear layer for final output.
    """

    @staticmethod
    def supported_model_types():
        return ["qwen3_moe"]

    def __init__(self, model_args: Qwen3MoeArgs):
        super().__init__(model_args.hf_config)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.rotary_emb = RotaryEmbedding(model_args)

        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = Qwen3MoEBlock(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )

        if not model_args.hf_config.tie_word_embeddings:
            self.tie_embed_tokens = False
            self.lm_head = nn.Linear(
                model_args.dim,
                model_args.vocab_size,
                bias="lm_head" in model_args.biases,
            )
        else:
            self.tie_embed_tokens = True
        self.identity_layer = IdentityLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        interested_tokens: Optional[torch.BoolTensor] = None,
        *args,
        **kwargs,
    ):
        if self.embed_tokens is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Do not remove this line
            # This is a trick for TP with torch.compile
            h = self.identity_layer(inputs_embeds)
        else:
            inputs_embeds = input_ids
            h = input_ids

        position_embeddings = self.rotary_emb(h, position_ids.to(dtype=torch.long))

        if "valid_input_len" in kwargs:
            valid_input_len = kwargs["valid_input_len"]
            updated_kwargs = pack_sequences_for_inputs(
                inputs_embeds,
                valid_input_len,
                list(position_embeddings),
                interested_tokens,
                inputs_seq_dim=1,
                inputs_batch_dim=0,
                position_ids_seq_dim=1,
                position_ids_batch_dim=0,
                interested_tokens_seq_dim=1,
                interested_tokens_batch_dim=0,
                padding_mask=kwargs.get("padding_mask", None),
                cp_mesh=kwargs.get("cp_mesh", None),
            )
            position_embeddings = tuple(updated_kwargs.pop("position_ids"))
            interested_tokens = updated_kwargs.pop("interested_tokens")
            h = updated_kwargs.pop("inputs")
            h = self.identity_layer(h)
            kwargs.update(updated_kwargs)

        for layer in self.layers.values():
            if (
                hasattr(layer, "_gradient_checkpointing_enabled")
                and layer._gradient_checkpointing_enabled
            ):
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    position_embeddings,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                h = layer(h, position_embeddings=position_embeddings, **kwargs)

        # Add `if` check just in case `pp` is enabled
        if self.norm is not None:
            if interested_tokens is not None:
                assert not isinstance(
                    h, torch.distributed.tensor.DTensor
                ), "interested_tokens must be a local tensor"
                h = h[interested_tokens]
            h = self.norm(h)
            if not self.tie_embed_tokens:
                output = self.lm_head(h)
            else:
                is_w_dist_tensor = isinstance(
                    self.embed_tokens.weight, torch.distributed.tensor.DTensor
                )
                embed_tokens_weight = (
                    self.embed_tokens.weight.full_tensor()
                    if is_w_dist_tensor
                    else self.embed_tokens.weight
                )
                is_a_dist_tensor = isinstance(h, torch.distributed.tensor.DTensor)
                h = h.full_tensor() if is_a_dist_tensor else h
                output = h @ embed_tokens_weight.t()
            return output
        else:
            return h

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        for layer in self.layers.values():
            layer.mlp.gate.weight.requires_grad_(False)

        # rotary.inv_freq could get deleted and not re-initialized
        # so we need to delete it manually
        self.rotary_emb.to(torch.cuda.current_device())
        self.rotary_emb.reset_inv_freq()
        # Basically, max_seq_len * 2 is enough for all-to-all-v communication.
        overflow = 2

        # TODO(cjx): max_seq_len * mini_batch is a better choice
        MAX_BATCH_MUL_SEQ_LEN = (
            self.model_args.max_seq_len
            * cosmos_config.train.train_batch_per_replica
            * self.model_args.hf_config.num_experts_per_tok
        )

        OnDeviceAllToAllV.max_output_len = MAX_BATCH_MUL_SEQ_LEN * overflow
        # Init MoE kernel related buffers
        if FeedForward.token_send_buf is None:
            dtype = self.model_args.hf_config.torch_dtype

            # Input buffer for DP-to-EP shuffle
            FeedForward.token_send_buf = symm_mem.empty(
                MAX_BATCH_MUL_SEQ_LEN,
                self.model_args.dim,  # hidden dim
                dtype=dtype,
                device=self.current_device(),
            )
            FeedForward.token_send_buf.zero_()
            # Input buffer for EP-to-DP shuffle
            FeedForward.token_gather_buf = symm_mem.empty(
                MAX_BATCH_MUL_SEQ_LEN * overflow,
                self.model_args.dim,  # hidden dim
                dtype=dtype,
                device=self.current_device(),
            )
            FeedForward.token_gather_buf.zero_()

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.qwen3_moe.parallelize import parallelize

        return parallelize, self

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert pp_size > 1
        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1

        # Compute the layers belonging to this stage
        n_layers = len(self.layers)
        layers_per_stage = n_layers // pp_size

        if not is_first:
            self.embed_tokens = None
        if not is_last:
            self.lm_head = None
            self.norm = None

        local_layers = torch.nn.ModuleDict()
        for i in range(
            pp_rank * layers_per_stage,
            ((pp_rank + 1) * layers_per_stage) if not is_last else n_layers,
        ):
            local_layers[str(i)] = self.layers[str(i)]

        # Reset the layers for pipeline splitting
        self.layers = local_layers

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
            info_inly (bool): Only collect the tensor infomation without actual data loading.
        """
        # Load all safetensors from `model_path`
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        self_state_dict = self.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}
        lm_head_weight_key = "lm_head.weight"
        embed_tokens_weight_key = "model.embed_tokens.weight"
        weights_of_ckpt_names = set()
        reserved = {}
        for f in safetensors_files:
            weights_of_ckpt = {}
            ckpt = safe_open(
                os.path.join(model_path, f), framework="pt", device=str(device)
            )
            keys = ckpt.keys()
            for name in keys:
                ckpt_tensor = ckpt.get_tensor(name)
                weights_of_ckpt[name] = ckpt_tensor
                weights_of_ckpt_names.add(name)
                if name == embed_tokens_weight_key:
                    reserved[name] = ckpt_tensor

            for name in weights_of_ckpt.keys():
                tensor = weights_of_ckpt[name]
                dest_name, shared_weight = convert_weight_from_hf(
                    tensor,
                    name,
                    model_type,
                    parallel_dims,
                    n_experts=self.model_args.n_experts,
                )

                if dest_name is None:
                    # This is due to the expert parallelism grouping
                    continue

                expert_id = None
                if match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)",
                    dest_name,
                ):
                    # remove `experts.$ID.` from dest_name
                    expert_id = int(match.group(2))
                    dest_name = dest_name.replace(f"experts.{expert_id}.", "")
                    # Convert expert_id to local_expert_id
                    n_local_experts = (
                        self.model_args.n_experts
                        // parallel_dims.tp
                        // parallel_dims.dp_shard
                    )

                    expert_id = expert_id % n_local_experts

                if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                    logger.info(
                        f"Weight `{dest_name}` is discarded, maybe due to pipeline parallelism or expert parallelism grouping. Skipping this weight checking"
                    )
                    continue

                target_tensor = self_state_dict[dest_name]
                if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                    target_tensor = target_tensor.to_local()
                # Write to the correct expert of the target tensor
                if expert_id is not None:
                    target_tensor = target_tensor[expert_id]

                assert (
                    target_tensor.shape == shared_weight.shape
                ), f"Shape mismatch: {target_tensor.shape} != {shared_weight.shape} for {dest_name}"
                with torch.no_grad():
                    target_tensor.data.copy_(shared_weight)

        if (
            lm_head_weight_key not in weights_of_ckpt_names
            and embed_tokens_weight_key in weights_of_ckpt_names
        ):
            # tied with embed_tokens.weight
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
                    local_view.data.copy_(shared_weight)

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, int]:
        seq_dim_idx = 1
        inputs = kwargs["input_ids"]
        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, seq_dim_idx

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]

    @cached_property
    def _get_nparams_and_flops_fn(self) -> Callable[[int], tuple[int, int]]:
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
            len(self.layers),
            self.model_args.n_heads,
            self.model_args.dim // self.model_args.n_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        return self._get_nparams_and_flops_fn(seq_len)

    @classmethod
    def from_model_args(cls, model_args: Qwen3MoeArgs) -> "Qwen3MoE":
        """
        Initialize a Qwen3Moe model from a Qwen3MoeArgs object.

        Args:
            model_args (Qwen3MoeArgs): Model configuration arguments.

        Returns:
            Qwen3MoE: Qwen3MoE model.

        """
        return cls(model_args)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "Qwen3MoE":
        """
        Initialize a Qwen3MoE model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            Qwen3MoE: Qwen3MoE model.

        """
        try:
            if hf_config.model_type not in cls.supported_model_types():
                raise ValueError(f"Unsupported model type: {hf_config.model_type}")

            if max_position_embeddings is None:
                max_position_embeddings = hf_config.max_position_embeddings
            else:
                hf_config.max_position_embeddings = max_position_embeddings

            vocab_size = sync_model_vocab(model_name_or_path)
            rope_scaling = {}
            if hasattr(hf_config, "rope_scaling"):
                rope_scaling = hf_config.rope_scaling or {}
            rope_type = rope_scaling.get("rope_type", "default")

            # Qwen3MoE does not have any biases
            bias_list = []
            try:
                head_dim = hf_config.head_dim
            except Exception:
                head_dim = hf_config.hidden_size // hf_config.num_attention_heads
                logger.warning(f"head_dim not found in config, using {head_dim}")

            model = cls.from_model_args(
                Qwen3MoeArgs(
                    dim=hf_config.hidden_size,
                    ffn_dim=hf_config.moe_intermediate_size,
                    n_layers=hf_config.num_hidden_layers,
                    n_experts=hf_config.num_experts,
                    n_heads=hf_config.num_attention_heads,
                    n_kv_heads=hf_config.num_key_value_heads,
                    head_dim=head_dim,
                    vocab_size=vocab_size,
                    max_seq_len=max_position_embeddings,
                    rope_theta=hf_config.rope_theta,
                    q_k_norm_enabled=hf_config.model_type == "qwen3_moe",
                    norm_type="rmsnorm",
                    rope_type=rope_type,
                    biases=bias_list,
                    hf_config=hf_config,
                )
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        return model

    @classmethod
    def fqn_filter_for_fp8(cls) -> List[str]:
        return ["lm_head"]

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        if not (self.model_args.n_heads % (cp_size * tp_size) == 0):
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's head number={self.model_args.n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )

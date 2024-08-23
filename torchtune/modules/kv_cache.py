# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn, Tensor


class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.size = 0
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor) -> Tuple[Tensor, Tensor]:
        """Update KV cache with the new k_val, v_val and return the updated cache.

        Raises an assertion error if ``input_pos`` is longer than the maximum sequence length.

        Args:
            input_pos (Tensor): Current position tensor with shape [S]
            k_val (Tensor): Current key tensor with shape [B, H, S, D]
            v_val (Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[Tensor, Tensor]: Updated KV cache with key first
        """
        import pdb

        # pdb.set_trace()
        assert input_pos.shape[-1] == k_val.shape[2]
        self.size = input_pos.dim()

        k_out = self.k_cache
        v_out = self.v_cache
        _, num_heads, _, d_k = k_out.shape
        # expanded_input_pos = input_pos.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, d_k).to(torch.long)

        # Use scatter to place k_val into k_out according to input_pos
        # k_out.scatter_(2, expanded_input_pos, k_val)
        # v_out.scatter_(2, expanded_input_pos, v_val)

        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

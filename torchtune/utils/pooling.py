# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_last_unmasked_token_idx(mask: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Returns the index for the last unmasked entry for each row of a 2D boolean mask.
    Args:
        mask (torch.Tensor): Boolean mask with shape [b x s], where True indicates a value to be masked out
            - this is usually a mask for padding tokens, where True indicates a padding token
        dtype (torch.dtype): dtype to cast the returned idxs to
    Returns:
        Tensor: Sequence indexes logits with shape [b]
    Notation used for tensor shapes:
        - b: batch size
        - s: sequence length

    Example:
        >>> input_ids = torch.tensor([
        >>>        [2, 4, 0, 0],
        >>>        [2, 4, 6, 0],
        >>>        [2, 4, 6, 9]
        >>>    ])
        >>> get_last_unmasked_token_idx(input_ids == 0)
        >>> tensor([1, 2, 3])
    """
    # calculate per-batch-element sequence lengths by finding last valid tokens
    if mask.any():
        sequence_lengths = (~mask).sum(-1).sub(1).clip(0).to(mask.device, dtype=dtype)
    else:
        sequence_lengths = torch.full((mask.shape[0],), mask.shape[1] - 1, dtype=dtype, device=mask.device)

    return sequence_lengths

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._generation import (
    generate_next_token_with_logits,
    generate_with_logits,
    get_causal_mask,
)
from .collate import left_padded_collate
from .kl_controller import AdaptiveKLController, FixedKLController
from .rewards import (
    estimate_advantages,
    get_reward_penalty_mask,
    get_rewards,
    masked_mean,
    masked_var,
    masked_whiten,
    whiten,
)
from .sequence_processing import (
    logits_to_logprobs,
    query_response_logits_to_response_logits,
    truncate_sequence_at_first_stop_token,
)

__all__ = [
    "generate_with_logits",
    "generate_next_token_with_logits",
    "truncate_sequence_at_first_stop_token",
    "get_causal_mask",
    "logits_to_logprobs",
    "query_response_logits_to_response_logits",
    "get_reward_penalty_mask",
    "update_stop_tokens_tracker",
    "left_padded_collate",
    "AdaptiveKLController",
    "FixedKLController",
    "estimate_advantages",
    "get_rewards",
    "whiten",
    "masked_whiten",
    "masked_mean",
    "masked_var",
]

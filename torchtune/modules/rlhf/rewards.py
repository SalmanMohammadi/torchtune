# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch


def get_reward_penalty_mask(
    padding_masks: torch.Tensor,
    seq_lens: torch.Tensor,
    penalise_no_eos: bool = True,
    min_response_length: bool = None,
) -> torch.Tensor:
    """
    Calculates a mask to penalise scores corresponding to sequences generated during PPO.
    Scores are penalised s.t.:
    - scores for sequences with length < min_response_length are set to -1
    - scores for sequences with no EOS token are set to -1

    Args:
        padding_masks (torch.Tensor): Tensors where True indicates a padding token in the generated
            sequence, and False otherwise. These are created when generated sequences are truncated at EOS token(s)
            and filled with padding values. Shape: (b, reponse_len)
        seq_lens (torch.Tensor): The length of each generated sequence. Shape: (b,)
        penalise_no_eos (bool): Whether to penalise sequences with no EOS token. Defaults to True.
        min_response_length (int, optional): The minimum length of the response. If set, any responses is shorter
            than this length
             will be penalised. Defaults to None.
    """
    reward_penalty_mask = torch.zeros_like(seq_lens).to(bool)

    if penalise_no_eos:
        reward_penalty_mask = ~padding_masks.any(-1)
    # - sequences with length < min_response_length recieve a score of -1
    if min_response_length is not None:
        reward_penalty_mask |= ~(seq_lens >= min_response_length)
    return reward_penalty_mask


def get_rewards(
    scores: torch.Tensor,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_controller_value: float,
    valid_score_idxs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the rewards for the given scores, logprobs, and reference logprobs.

    Args:
        scores (torch.Tensor): Reward model scores, shape (b,).
        logprobs (torch.Tensor): Policy logprobs, shape (b, reponse_len).
        ref_logprobs (torch.Tensor): Reference base model, shape (b, reponse_len).
        kl_controller_value (float): Adaptive KL controller value.
        valid_score_idxs (torch.Tensor, optional): A tensor of indexes for valid (non-padded) token predictions.
            This is useful when calculating rewards for padded sequences, as scores and value estimates are defined
            for the last valid predicted token. Shape: (b,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors with shape [b, response_len] each:
            - total_reward
            - kl between policy and reference base model
            - reward corresponding to kl above

    Notation used for tensor shapes:
        - b: batch size
        - response_len: model response length
    """

    # 1. calculate kl between logprobs and reflogprobs
    # 2. calculate kl reward using adaptive scaling value
    # 3. calculate total reward by summing above
    # return all
    kl = logprobs - ref_logprobs
    kl_reward = -kl_controller_value * kl

    total_reward = kl_reward.clone()

    # adding reward to kl at final valid position
    # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L153

    if valid_score_idxs is not None:
        total_reward[
            torch.arange(scores.shape[0], device=scores.device), valid_score_idxs
        ] += scores
    else:
        total_reward[:, -1] += scores

    return total_reward, kl, kl_reward


def whiten(x: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """
    Whitens (normalizes) the input tensor.

    Args:
        advantages (torch.Tensor): The advantages.

    Returns:
        torch.Tensor: The whitened tensor.
    """
    mean, var = x.mean(), x.var()
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if shift_mean:
        whitened += mean
    return whitened


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Compute mean of tensor with masked values. Taken from https://github.com/huggingface/trl/blob/main/trl/core.py

    Args:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): The bool mask tensor, where True indicates the corresponding value in `x`
            should participate in the mean calculation.
        dim (int): The axis to calculate the mean over.

    Returns:
        torch.Tensor: The mean tensor.
    """
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_var(
    x: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """
    Compute variance of tensor with masked values. Taken from https://github.com/huggingface/trl/blob/main/trl/core.py

    Args:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): The bool mask tensor, where True indicates the corresponding value in `x`
            should participate in the mean calculation.
        unbiased (bool): Whether to use the unbiased variance.

    Returns:
        torch.Tensor: The variance tensor.

    Raises:
        ValueError: If the sum of the mask is zero.
    """
    mean = masked_mean(x, mask)
    centered_values = x - mean
    var = masked_mean(centered_values.pow(2), mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `ppo_batch_size=1`;"
                "try increase the `ppo_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        var = var * bessel_correction
    return var


def masked_whiten(
    x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """
    Whiten (normalises) values with masked values. Taken from https://github.com/huggingface/trl/blob/main/trl/core.py
    Args:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): The bool mask tensor, where True indicates the corresponding value in `x`
            should participate in the mean calculation.
        shift_mean (bool): Whether to shift normalised values by the mean.

    Returns:
        torch.Tensor: The whitened tensor.
    """
    mean = masked_mean(x, mask)
    var = masked_var(x, mask) if mask.any() else x.var()
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if shift_mean:
        whitened += mean
    return whitened


def estimate_advantages(
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lmbda: float,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the advantages and returns for the PPO algorithm using Generalized Advantage Estimation
    https://arxiv.org/pdf/1506.02438.pdf.

    Args:
        values (torch.Tensor): The predicted values for each state. Shape: (b, reponse_len)
        rewards (torch.Tensor): The rewards received at each time step. Shape: (b, reponse_len)
        gamma (float): The discount factor.
        lmbda (float): The GAE-Lambda parameter.
        masks (torch.Tensor, optional): A bool mask tensor, where True indicates the corresponding value in `values`
            should participate in the mean calculation.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the estimated advantages and returns.
            - advantages (torch.Tensor): The estimated advantages. Shape: (b, reponse_len)
            - returns (torch.Tensor): The estimated returns. Shape: (b, reponse_len)
    Notation:
        - b: batch size
        - reponse_len: model response length
    """

    last_gae_lam = 0
    advantages_reversed = []

    response_length = values.shape[-1]

    # estimate advantage for every predicted token position
    for t in reversed(range(response_length)):
        # value of the next state
        next_values = values[:, t + 1] if t < response_length - 1 else 0.0
        # exponentially discounted temporal difference error:
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        # GAE-Lambda advantage discouting saved for the next iteration
        # as A_t = delta_t + gamma * lambda * A_{t+1} + ...
        last_gae_lam = delta + gamma * lmbda * last_gae_lam
        advantages_reversed.append(last_gae_lam)

    advantages = torch.stack(advantages_reversed[::-1], axis=1)

    # returns are the expected value of taking action a_t at each timepoint over
    # a trajectory. the value estimates v_t are the expected value over all actions
    # over a trajectory - the advantage is the difference between the two
    returns = advantages + values

    # normalize advantages across the batch of trajectories to reduce variance
    if masks is not None:
        advantages = masked_whiten(advantages, masks)
    else:
        advantages = whiten(advantages)

    return advantages, returns

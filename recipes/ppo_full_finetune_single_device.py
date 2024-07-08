# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from functools import partial
from itertools import chain
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from pkg_resources import packaging
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.datasets import ConcatDataset
from torchtune.modules import rlhf
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm

log = utils.get_logger("DEBUG")

Trajectory = NamedTuple(
    "Trajectory",
    [
        ("query_responses", torch.Tensor),
        ("logprobs", torch.Tensor),
        ("ref_logprobs", torch.Tensor),
        ("values", torch.Tensor),
        ("masks", torch.Tensor),
        ("position_ids", torch.Tensor),
        ("response_padding_masks", torch.Tensor),
        ("value_padding_masks", torch.Tensor),
        ("value_seq_idxs", torch.Tensor),
        ("scores", torch.Tensor),
    ],
)


class PPOFullFinetuneRecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:

        # self._device = utils.get_device(device=cfg.device)
        self._device = torch.device("mps")
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError("fp16 precision is not supported in this recipe. Please use fp32 or bf16.")
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if self._dtype == torch.bfloat16 and self._device != torch.device("cpu"):
            if torch.cuda.is_available():
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Full bf16 training is not supported on this hardware.")
            elif torch.backends.mps.is_available():
                if packaging.version.parse(torch.__version__).release < (2, 3):
                    raise RuntimeError("Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = 0
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                }
            )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._policy_model.state_dict().items()}

        # save base model as usual
        ckpt_dict.update({utils.MODEL_KEY: state_dict})

        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {k: v for k, v in self._policy_model.state_dict().items() if adapter_key_filter(k)}
        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({utils.ADAPTER_CONFIG: adapter_config})
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

        self._policy_checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed or total_epoch,
        # warn the user and overwrite
        if self.seed != ckpt_dict[utils.SEED_KEY] or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]:
            warn(
                message="""Configured value for seed or epochs
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # setup checkpointers
        self._policy_checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        self._reward_checkpointer = config.instantiate(
            cfg.reward_checkpointer,
            # reward model is never being trained
            # base model checkpointer handles recipe state
            resume_from_checkpoint=False,
        )
        # load base model checkpoint
        policy_model_checkpoint_dict = self._policy_checkpointer.load_checkpoint()

        # load reward model checkpoint
        reward_model_checkpoint_dict = self._reward_checkpointer.load_checkpoint()

        # update recipe state
        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        (
            self._policy_model,
            self._value_model,
            self._reward_model,
            self._ref_policy_model,
        ) = self._setup_model(
            cfg_model=cfg.policy,
            cfg_reward_model=cfg.reward_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            model_state_dict=policy_model_checkpoint_dict[utils.MODEL_KEY],
            reward_model_state_dict=reward_model_checkpoint_dict[utils.MODEL_KEY],
        )

        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # setup opt
        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(policy_model_checkpoint_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None),
        )

        # GAE hyperparameters
        self.gamma = cfg.loss.gamma
        self.lmbda = cfg.loss.lmbda
        self.whiten_rewards = cfg.whiten_rewards
        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # sampler and dataloader depends on the tokenizer and should be set
        # setup afterit is initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # setup training params:
        # num_steps - global number of optimisation steps (batches)
        # batch_size - number of samples in a batch
        # ppo_epochs - number of epochs to optimise the policy over a batch of episodes
        # ppo_batch_size - number of minibatches (sampled from a single batch) to optimise the policy over
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        if self.ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self.ppo_batch_size})  must be "
                f"exactly divisible by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )
        self.ppo_backward_batch_size = cfg.ppo_batch_size // self._gradient_accumulation_steps

        # trajectory generation args
        self.temperature = cfg.temperature + 1e-7  # avoid underflow for 0 temperature
        self.top_k = cfg.top_k
        self.forward_batch_size = cfg.forward_batch_size
        self.max_generated_tokens = cfg.max_generated_tokens

        # reward masking args
        self.min_response_length = cfg.min_response_length
        self.penalise_no_eos = cfg.penalise_no_eos
        self.reward_penalty = cfg.reward_penalty

        # lots of hand holding for stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                log.warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer.stop_tokens):
                log.warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens

        self.stop_token_ids = torch.tensor(stop_token_ids, device=self._device)
        self.total_epochs = self.num_steps // self.batch_size
        # one "step" is an update over a batch of trajectories
        self._steps_per_epoch = self.num_steps  # len(self._dataloader) // self.batch_size
        self.global_step = self.epochs_run * self._steps_per_epoch

        # setup adaptive KL controller
        self.kl_controller = config.instantiate(cfg.kl_controller)

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        cfg_reward_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:

        with utils.set_default_dtype(self._dtype), self._device:
            policy_model = config.instantiate(cfg_model)
            ref_policy_model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_model)
            value_model = config.instantiate(cfg_reward_model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(policy_model, auto_wrap_policy={modules.TransformerDecoderLayer})
            utils.set_activation_checkpointing(value_model, auto_wrap_policy={modules.TransformerDecoderLayer})

        reward_missing, reward_unexpected = reward_model.load_state_dict(reward_model_state_dict, strict=False)
        value_missing, value_unexpected = value_model.load_state_dict(reward_model_state_dict, strict=False)

        assert (
            reward_missing == value_missing == []
        ), f"Missing keys in reward ({reward_missing}) and value model ({value_missing}) state dicts."

        if reward_unexpected or value_unexpected:
            assert (
                reward_unexpected == value_unexpected == ["output.bias"]
            ), f"Unexpected keys in reward ({reward_unexpected}) and value model ({value_unexpected}) state dicts."

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(value_model.named_parameters(), dtype=self._dtype)
        log.info(f"Base model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(reward_model.named_parameters(), dtype=self._dtype)
        log.info(f"Reward model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(value_model.named_parameters(), dtype=self._dtype)
        log.info(f"Value model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(ref_policy_model.named_parameters(), dtype=self._dtype)
        log.info(f"Ref model is initialized with precision {self._dtype}.")

        # disabling dropout if found
        for model in [policy_model, value_model, ref_policy_model, reward_model]:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0

        # Compile model, if enabled.
        if compile_model:
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

            log.info("Compiling policy model with torch.compile...")
            policy_model.compile(backend=backend)

            log.info("Compiling reward model with torch.compile...")
            reward_model.compile(backend=backend)

            log.info("Compiling ref model with torch.compile...")
            ref_policy_model.compile(backend=backend)

            log.info("Compiling value model with torch.compile...")
            value_model.compile(backend=backend)

        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        reward_model.eval()
        ref_policy_model.eval()

        return policy_model, value_model, reward_model, ref_policy_model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:

        # not sure if chain is necessary?
        optimizer = config.instantiate(
            cfg_optimizer,
            chain(self._policy_model.parameters(), self._value_model.parameters()),
        )
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer) for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                rlhf.left_padded_collate,
                padding_idx=self._tokenizer.pad_id,
            ),
            drop_last=True,
        )

        def repeat_generator(dataloader: DataLoader) -> Iterator[torch.Tensor]:
            # infinite generator for when num_steps > len(dataloader) // self.batch_size
            while True:
                yield from dataloader

        log.info("Dataset and Sampler are initialized.")

        return sampler, iter(repeat_generator(dataloader))

    def generate_trajectory(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a trajectory given the current policy and value model and batch of inputs.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory (NamedTuple): A collection of tensors corresponding to the current trajectory:
            - query_responses (torch.Tensor): generated responses with shape [b, max_generated_tokens]
            - logprobs (torch.Tensor): log probabilities of the generated responses with shape [b, max_generated_tokens]
            - ref_logprobs (torch.Tensor): log probabilities of the generated responses using the reference policy with shape [b, max_generated_tokens]
            - values (torch.Tensor): value estimates of the generated responses with shape [b, max_generated_tokens]
            - masks (Any[torch.Tensor]): attention masks with shape [b, max_generated_tokens, max_generated_tokens]
            - position_ids (Any[torch.Tensor]): position IDs with shape [b, max_generated_tokens]
            - response_padding_masks (torch.Tensor): padding masks for the post-processed, truncated generated responses with shape [b, max_generated_tokens]
            - value_padding_masks (torch.Tensor): padding masks for the values with shape [b, max_generated_tokens]
            - value_seq_idxs (torch.Tensor): indexes of the last valid (non-padding) token in the response with shape [b]
            - scores (torch.Tensor): scores from the reward model with shape [b]

        """
        batch_size, context_length = input_ids.shape

        # step 1: generate responses, and logits corresponding to the responses using the policy
        query_responses, logits = rlhf.generate_with_logits(
            model=self._policy_model,
            prompt=input_ids,
            max_generated_tokens=self.max_generated_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            pad_id=self._tokenizer.pad_id,
        )
        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses == self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = rlhf.get_causal_mask(~(query_response_padding_masks))
        position_ids = (~query_response_padding_masks).cumsum(-1) - (~query_response_padding_masks).long()
        position_ids = position_ids.type(torch.int)

        del query_response_padding_masks

        # step 2. estimate logprobs of the generated responses using the current policy
        logits = logits[:, context_length - 1 :]  # [b, max_generated_tokens, vocab_size]
        logprobs = rlhf.logits_to_logprobs(logits, responses, self.temperature)

        del logits

        # step 2.1 estimate logprobs of the generated responses using the reference policy
        ref_logits = self._ref_policy_model(query_responses, input_pos=position_ids, mask=masks)
        ref_logits = rlhf.query_response_logits_to_response_logits(ref_logits, context_length)
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self.temperature)

        del ref_logits

        import pdb

        pdb.set_trace()
        # step 3. estimate values from the generated responses using the value function
        values = self._value_model(query_responses, input_pos=position_ids, mask=masks)
        values = rlhf.query_response_logits_to_response_logits(values, context_length).squeeze(
            -1
        )  # [b, max_generated_tokens]

        # step 4. replace any tokens in the response after the first stop token (usually EOS token) with padding
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, self.stop_token_ids, self._tokenizer.pad_id
        )

        # step 5. run the reward model on the query truncated-response pairs
        scores = self._reward_model(
            torch.cat([input_ids, responses], dim=1),
            input_pos=position_ids,
            mask=masks,
        )

        del responses

        # step 5.1 the reward scores from the reward model are the logits for the last non-padding token in each query+response pair
        seq_lens = utils.get_unmasked_sequence_lengths(response_padding_masks)
        scores = scores[torch.arange(batch_size), seq_lens + context_length].squeeze(-1)

        # step 5.2 if configured, apply any penalties for sequences without EOS tokens or shorter than a certain length
        if self.penalise_no_eos or self.min_response_length:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                response_padding_masks,
                seq_lens,
                self.penalise_no_eos,
                self.min_response_length,
            )
            scores[reward_penalty_mask] = self.reward_penalty

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        # step 6.1 values are masked out *after* the last valid token in the response
        value_seq_idxs = torch.where(
            (seq_lens > 0) & (seq_lens < self.max_generated_tokens - 1),
            seq_lens + 1,
            seq_lens,
        )
        value_padding_masks = response_padding_masks.clone()
        value_padding_masks[
            torch.arange(batch_size, device=value_padding_masks.device),
            value_seq_idxs,
        ] = False

        values[value_padding_masks] = 0.0

        return Trajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            value_padding_masks=value_padding_masks,
            value_seq_idxs=value_seq_idxs,
            scores=scores,
        )

    def generate_trajectory_batched(self, input_ids: torch.Tensor) -> Trajectory:
        trajectories: List[Trajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self.forward_batch_size):
                batch_input_ids = input_ids[batch_start : batch_start + self.forward_batch_size]
                trajectories.append(self.generate_trajectory(batch_input_ids))
        return Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop."""
        pbar = tqdm(total=self.total_epochs)
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            with self._profiler:

                if self._profiler_enabled:
                    self._profiler.step()

                batch = next(self._dataloader).to(self._device)
                _, context_length = batch.shape

                # step 1. generate the trajectory using:
                # - the current policy (pi_theta)
                # - the current value function (V_phi)
                # - the reference frozen policy model (pi_theta_0)
                trajectory = self.generate_trajectory_batched(batch)

                # step 2. get the rewards for the current trajectory. these are based on:
                #   - the divergence between the current policy and the reference policy
                #   - the scores from the reward model
                rewards, kl, kl_rewards = rlhf.get_rewards(
                    trajectory.scores,
                    trajectory.logprobs,
                    trajectory.ref_logprobs,
                    self.kl_controller.value,
                    trajectory.value_seq_idxs,
                )

                # step 3. estimate the advantages using Generalized Advantage Estimation (GAE)
                advantages, returns = rlhf.estimate_advantages(
                    trajectory.values,
                    rewards,
                    self.gamma,
                    self.lmbda,
                    masks=~trajectory.response_padding_masks,
                )
                advantages[trajectory.response_padding_masks] = 0.0

                # step 4. optimise using the PPO objective over multiple epochs
                ppo_stats = [
                    [],
                ] * 6
                for epoch_idx in range(self.ppo_epochs):
                    batch_idxs = torch.randperm(self.batch_size, device=self._device)
                    for i in range(0, self.batch_size, self.ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]

                        batch_ppo_stats = [0] * 6
                        for j in range(0, self.ppo_batch_size, self.ppo_backward_batch_size):
                            backward_batch_idxs = mini_batch_idxs[j : j + self.ppo_backward_batch_size]
                            batch_trajectory = Trajectory(
                                *map(
                                    partial(
                                        torch.index_select,
                                        dim=0,
                                        index=backward_batch_idxs,
                                    ),
                                    trajectory,
                                )
                            )

                            backward_ppo_stats = self.ppo_step(
                                batch_trajectory,
                                advantages[backward_batch_idxs],
                                returns[backward_batch_idxs],
                                context_length,
                            )

                            batch_ppo_stats = [
                                stat + bstat for stat, bstat in zip(batch_ppo_stats, backward_ppo_stats)
                            ]

                            del batch_trajectory

                        ppo_stats = [
                            ppo_stat + [batch_ppo_stat] for ppo_stat, batch_ppo_stat in zip(ppo_stats, batch_ppo_stats)
                        ]
                        self._optimizer.step()
                        self._optimizer.zero_grad()
                        self.global_step += 1

                # step 5. profit

            ppo_stats = map(torch.tensor, ppo_stats)
            self.log_metrics(
                trajectory,
                kl,
                kl_rewards,
                *ppo_stats,
            )
            self.kl_controller.update(kl.sum(1).mean().item(), self.global_step)
            self.cleanup_step(trajectory, advantages, returns, kl, kl_rewards)
            pbar.update(1)

    def ppo_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        context_length: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Perform a single PPO optimisation step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            advantages (torch.Tensor): advantages corresponding to the trajectories
            returns (torch.Tensor): returns corresponding the trajectories
            context_length (int): input ids sequence length

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                loss: Actor-critic combined policy and value loss
                policy_loss: PPO policy loss
                value_loss: Clipped critic value loss
                approx_policy_kls: Average estimated KL divergence between the policy before and after the optimisation step
                ratios: Logprob ratios between the current policy being optimised, and the policy used to generate the trajectory
                clipfrac: Fraction of samples where the policy ratio has been clipped in the PPO objective
        """
        # estimate logprobs from the policy at the current optimisation step
        pi_logits = self._policy_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )
        pi_logits = rlhf.query_response_logits_to_response_logits(pi_logits, context_length)
        pi_logprobs = rlhf.logits_to_logprobs(
            pi_logits, trajectory.query_responses[:, context_length:], self.temperature
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0

        # sesimate the values from the value function at the current optimisation step
        phi_values = self._value_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )

        phi_values = rlhf.query_response_logits_to_response_logits(phi_values, context_length).squeeze(-1)
        phi_values[trajectory.value_padding_masks] = 0.0

        # calculate ppo loss
        loss, policy_loss, value_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            advantages,
            trajectory.values,
            phi_values,
            returns,
            padding_masks=~trajectory.response_padding_masks,
            value_padding_masks=~trajectory.value_padding_masks,
        )

        loss /= self._gradient_accumulation_steps

        loss.backward()

        policy_loss /= self._gradient_accumulation_steps
        value_loss /= self._gradient_accumulation_steps

        with torch.no_grad():
            approx_policy_kls = (0.5 * (pi_logprobs - trajectory.logprobs).pow(2)).mean()

        return loss, policy_loss, value_loss, approx_policy_kls, ratios, clipfrac

    def log_metrics(
        self,
        trajectory: Trajectory,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        approx_policy_kls: torch.Tensor,
        ratios: torch.Tensor,
        clipfrac: torch.Tensor,
    ) -> None:
        """
        Log metrics and statistics for the current step to the metric logger.
        """
        log_dict = {
            "scores": trajectory.scores.mean(),
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "rlhf_reward": trajectory.scores.mean() + kl_rewards.sum(1).mean(),
            "kl": kl.sum(1).mean(),
            "kl_reward": kl_rewards.sum(1).mean(),
            "loss": loss.mean(),
            "policy_loss": policy_loss.mean(),
            "value_loss": value_loss.mean(),
            "approx_policy_kl": approx_policy_kls.mean(),
            "clipfrac": clipfrac.mean(),
            "ratios": ratios.mean(),
        }
        self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        # there shouldn't be any floating references to the individual tensors at the this point, so gc can do its thing
        for tensor in trajectory:
            del tensor
        del trajectory
        del advantages
        del returns
        del kl
        del kl_rewards

    def cleanup(self, **kwargs) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="PPOFullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = PPOFullFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

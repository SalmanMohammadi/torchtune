# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
from functools import partial
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
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

log = utils.get_logger("INFO")

Checkpointer = Union[
    utils.FullModelHFCheckpointer,
    utils.FullModelMetaCheckpointer,
    utils.FullModelTorchTuneCheckpointer,
]
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

        if cfg.device == "mps":
            self._device = torch.device("mps")
        else:
            self._device = utils.get_device(device=cfg.device)
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
        # just tanking the hit to re-initialise the torch generator to avoid modifying `set_seed`
        self._rng = torch.Generator(self._device).manual_seed(self.seed)
        self._total_steps = 0
        self._steps_run = 0
        self._total_epochs = 0
        self._epochs_run = 0
        self.global_step = 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def save_checkpoint(self, epoch: int, is_intermediate_checkpoint: bool = False) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        policy_ckpt_dict = {utils.MODEL_KEY: self._policy_model.state_dict()}
        value_ckpt_dict = {utils.MODEL_KEY: self._value_model.state_dict()}

        # if training is in-progress, checkpoint the optimizer state and rng state as well
        if is_intermediate_checkpoint:
            policy_ckpt_dict.update(
                {
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self._epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self._total_epochs,
                    utils.MAX_STEPS_KEY: self._total_steps,
                    utils.STEPS_KEY: self._steps_run,
                    utils.RNG_KEY: self._rng.get_state(),
                }
            )
            policy_ckpt_dict[utils.OPT_KEY] = self._optimizer.state_dict()

        self._policy_checkpointer.save_checkpoint(
            policy_ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=is_intermediate_checkpoint,
        )

        self._value_checkpointer.save_checkpoint(
            value_ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=False,
        )

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed or total_steps don't match,
        # warn the user and overwrite

        #
        try:
            if (
                self.seed != ckpt_dict[utils.SEED_KEY]
                or self._total_steps != ckpt_dict[utils.MAX_STEPS_KEY]
                or self._total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            ):
                warn(
                    message="""Configured value for seed, total_steps, or total_epochs
                    does not match the value stored in checkpoint."""
                )
            self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
            self._rng.set_state(ckpt_dict[utils.RNG_KEY])
            self._steps_run = ckpt_dict[utils.STEPS_KEY]
            self._total_steps = ckpt_dict[utils.MAX_STEPS_KEY]
            self._total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            self._epochs_run = ckpt_dict[utils.EPOCHS_KEY]

        except KeyError as e:
            raise KeyError from e(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # setup checkpointers
        (
            self._policy_checkpointer,
            ref_policy_checkpointer,
            self._value_checkpointer,
            reward_checkpointer,
        ) = self._setup_checkpointers(
            cfg.checkpointer,
            cfg.ref_policy_checkpointer,
            cfg.value_checkpointer,
            cfg.reward_checkpointer,
        )

        # load policy checkpoints
        policy_model_checkpoint_dict = self._policy_checkpointer.load_checkpoint()
        ref_policy_state_dict = ref_policy_checkpointer.load_checkpoint()[utils.MODEL_KEY]

        # load reward and value model checkpoints
        value_model_checkpoint_dict = self._value_checkpointer.load_checkpoint()[utils.MODEL_KEY]
        reward_model_state_dict = reward_checkpointer.load_checkpoint()[utils.MODEL_KEY]

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
            cfg_model=cfg.policy_model,
            cfg_reward_value_model=cfg.reward_and_value_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            policy_state_dict=policy_model_checkpoint_dict[utils.MODEL_KEY],
            ref_policy_state_dict=ref_policy_state_dict,
            value_model_state_dict=value_model_checkpoint_dict,
            reward_model_state_dict=reward_model_state_dict,
        )
        self.cfg_model = cfg.policy_model

        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=cfg.optimizer_in_bwd,
            opt_state_dict=(policy_model_checkpoint_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # sampler and dataloader depends on the tokenizer and should be set
        # setup afterit is initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        self._setup_training_parameters(cfg)
        self._setup_training_hyperparameters(cfg)

        if self._resume_from_checkpoint:
            self._update_recipe_state(policy_model_checkpoint_dict)

        # one "step" is a single gradient update update over a minibatch of trajectories
        self.global_step = self._steps_run * self._ppo_epochs * (self.batch_size // self._ppo_batch_size)
        self._optimizer_in_bwd = cfg.optimizer_in_bwd

    def _setup_training_hyperparameters(self, cfg) -> None:
        """
        Sets up the training hyperparameters for the recipe. This includes the GAE hyperparameters,
        generation hyperparameters, reward masking hyperparameters, and stop token ids.
        """

        self._kl_coeff = cfg.kl_coeff
        # GAE hyperparameters
        self._gamma = cfg.loss.gamma
        self._lmbda = cfg.loss.lmbda
        self._whiten_rewards = cfg.whiten_rewards

        # trajectory generation args
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens

        # reward masking args
        self._min_response_length = cfg.min_response_length
        self._penalise_no_eos = cfg.penalise_no_eos
        self._reward_penalty = cfg.reward_penalty

        # lots of hand holding for stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer.stop_tokens):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

    def _setup_training_parameters(self, cfg: DictConfig) -> None:
        """
        Validates and sets up parameters for used during training and for tracking training state,
        batch sizes for model forward passes during trajectory generation, PPO minibatches, and
        PPO microbatches for gradient accumulation.

        Raises
            - ValueError if:
                - batch_size is not divisible by forward_batch_size
                - batch_size is not divisible by ppo_batch_size
                - ppo_batch_size is not divisible by gradient_accumulation_steps
                - num_steps is less than batch_size
        """
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = cfg.ppo_batch_size // self._gradient_accumulation_steps

        if self.batch_size % self._forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self._forward_batch_size})."
            )
        if self.batch_size % self._ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self._ppo_batch_size})."
            )
        if self._ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self._ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )

        self._total_steps = cfg.num_steps // self.batch_size
        batches_per_epoch = len(self._dataloader)
        self._total_epochs = math.ceil(self._total_steps / batches_per_epoch)
        if self._total_steps == 0:
            raise ValueError(f"num_steps {cfg.num_steps} must be greater than the batch size {self.batch_size}.")
        if self._total_steps < len(self._dataloader):
            warn(
                f"There are fewer total steps ({self._total_steps} (num_steps//batch_size) "
                f"than there are batches ({len(self._dataloader)}) in the dataset. "
                f"Training will stop after ({self._total_steps}) steps without saving intermediate checkpoints"
            )
        if (self._total_steps > batches_per_epoch) and (self._total_steps % batches_per_epoch != 0):
            warn(
                f"num_steps ({cfg.num_steps}) is not exactly divisible by "
                f"the number of batches in the dataset ({batches_per_epoch}). "
                f"Intermediate checkpoints will only be saved every {batches_per_epoch} steps."
            )
        log.info(f"Total steps to run: {self._total_steps}, Total epochs to run: {self._total_epochs}")

    def _setup_checkpointers(
        self,
        policy_cfg: DictConfig,
        ref_policy_cfg: DictConfig,
        value_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> Tuple[Checkpointer, Checkpointer, Checkpointer, Checkpointer]:
        """
        Sets up checkpointers for policy, reference policy, value, and reward models.
        Only the policy checkpoint handles recipe state for resuming from checkpoints.
        """

        if not self._resume_from_checkpoint:
            assert policy_cfg.checkpoint_dir == ref_policy_cfg.checkpoint_dir, (
                "Policy and reference policy should be loaded from the same checkpoint directories"
                f"at the start of training. Found: {policy_cfg.checkpoint_dir} and"
                f"{ref_policy_cfg.checkpoint_dir}"
            )
            assert policy_cfg.checkpoint_files == ref_policy_cfg.checkpoint_files, (
                "Policy and reference policy should be loaded from the same checkpoint files"
                f"at the start of training. Found: {policy_cfg.checkpoint_files} and"
                f"{ref_policy_cfg.checkpoint_files}"
            )

        policy_checkpointer = config.instantiate(
            policy_cfg,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )

        ref_policy_checkpointer = config.instantiate(
            ref_policy_cfg,
            resume_from_checkpoint=False,
        )

        value_checkpointer = config.instantiate(
            value_cfg,
            resume_from_checkpoint=False,
        )

        reward_checkpointer = config.instantiate(
            reward_cfg,
            resume_from_checkpoint=False,
        )

        return (
            policy_checkpointer,
            ref_policy_checkpointer,
            value_checkpointer,
            reward_checkpointer,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        cfg_reward_value_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        policy_state_dict: Dict[str, Any],
        ref_policy_state_dict: Dict[str, Any],
        value_model_state_dict: Dict[str, Any],
        reward_model_state_dict: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:

        with utils.set_default_dtype(self._dtype), self._device:
            policy_model = config.instantiate(cfg_model)
            ref_policy_model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_value_model)
            value_model = config.instantiate(cfg_reward_value_model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(policy_model, auto_wrap_policy={modules.TransformerDecoderLayer})
            utils.set_activation_checkpointing(value_model, auto_wrap_policy={modules.TransformerDecoderLayer})

        policy_model.load_state_dict(policy_state_dict)
        ref_policy_model.load_state_dict(ref_policy_state_dict)

        reward_missing, reward_unexpected = reward_model.load_state_dict(reward_model_state_dict, strict=False)
        value_missing, value_unexpected = value_model.load_state_dict(value_model_state_dict, strict=False)

        # some extra validation for HF classifier checkpoints with a `score.bias` present
        assert (
            reward_missing == value_missing == []
        ), f"Missing keys in reward ({reward_missing}) and value model ({value_missing}) state dicts."

        if reward_unexpected or value_unexpected:
            # the only unexpected keys should be when pre-trained HF models were saved with
            # bias=True in final classification layers.
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

        for model in [policy_model, value_model]:
            # disabling dropout if found - non-determinism leads to issues in e.g. comparing logprobs
            # between ref policy and current policy
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    warn(f"Dropout found in {module}. This is likely to cause issues during training. Disabling.")
                    module.p = 0

        # Compile model, if enabled.
        if compile_model:
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            backend = "aot_eager"  # TODO (SalmanMohammadi)
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
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:

        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in chain(self._policy_model.parameters(), self._value_model.parameters())
            }
            # Register optimizer step hooks on the model to run optimizer in backward.
            utils.register_optim_in_bwd_hooks(model=self._policy_model, optim_dict=optim_dict)
            utils.register_optim_in_bwd_hooks(model=self._value_model, optim_dict=optim_dict)
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            # self._optim_ckpt_wrapper = utils.create_optim_in_bwd_wrapper(model=self._model, optim_dict=optim_dict)
            # self._optim_ckpt_wrapper = utils.create_optim_in_bwd_wrapper(
            #     model=self._value_model, optim_dict=optim_dict
            # )
            # # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # # use optimizer in backward.
            # if opt_state_dict is not None:
            #     try:
            #         self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
            #     except BaseException as e:
            #         raise RuntimeError(
            #             "Failed loading in-backward optimizer checkpoints."
            #             "Please make sure run being restored from was using in-backward optimizer."
            #         ) from e
            log.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(
                cfg_optimizer,
                chain(self._policy_model.parameters(), self._value_model.parameters()),
            )
            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)

            log.info("Optimizer is initialized.")
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

        return sampler, dataloader

    def get_new_model(self, sd=None):

        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(self.cfg_model)
        if sd is None:
            new_checkpointer = utils.FullModelHFCheckpointer(
                checkpoint_dir="./target/output_policy/",
                checkpoint_files=["hf_model_0001_0.pt"],
                output_dir="./target/tmp/",
                model_type="LLAMA2",
            )
            sd = new_checkpointer.load_checkpoint()["model"]
        model.load_state_dict(sd)
        return model

    def generate_trajectory(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a trajectory given the current policy and value models, the reference policy model, the reward model,
        and batch of inputs.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory (NamedTuple): A collection of tensors comprising the trajectory for the current minibatch:
            - query_responses (torch.Tensor): input ids-generated responses pairs
                shape [b, context_length + max_generated_tokens]
            - logprobs (torch.Tensor): log probabilities of the generated responses with shape [b, max_generated_tokens]
            - ref_logprobs (torch.Tensor): log probabilities of the generated responses using the reference policy
                shape [b, max_generated_tokens]
            - values (torch.Tensor): value estimates of the generated responses with shape [b, max_generated_tokens]
            - masks (torch.Tensor): attention masks for input ids-generated responses pairs
                shape [b, context_length + max_generated_tokens, context_length + max_generated_tokens]
            - position_ids (torch.Tensor): position IDs for input ids-generated responses pairs
                shape [b, context_length + max_generated_tokens]
            - response_padding_masks (torch.Tensor): padding masks for the truncated and padded generated responses
                shape [b, max_generated_tokens]
            - value_padding_masks (torch.Tensor): padding masks for the values with
                shape [b, max_generated_tokens]
            - value_seq_idxs (torch.Tensor): indexes of the token
                after the last valid (non-padding) token in the responses with shape [b]
            - scores (torch.Tensor): scores from the reward model with shape [b]

        """
        batch_size, context_length = input_ids.shape

        # step 1: generate responses, and logits corresponding to the responses using the policy
        query_responses, logits = rlhf.generate_with_logits(
            model=self._policy_model,
            prompt=input_ids,
            max_generated_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k,
            pad_id=self._tokenizer.pad_id,
            rng=self._rng,
        )

        #
        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses == self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = rlhf.get_causal_mask(~(query_response_padding_masks))
        position_ids = (~query_response_padding_masks).cumsum(-1) - (~query_response_padding_masks).long()
        position_ids = position_ids.type(torch.int)

        del query_response_padding_masks

        # step 2. estimate logprobs of the generated responses using the current policy
        logits = logits[:, context_length - 1 :]
        logprobs = rlhf.logits_to_logprobs(logits, responses, self._temperature)

        del logits

        # step 2.1 estimate logprobs of the generated responses using the reference policy
        ref_logits = self._ref_policy_model(query_responses, input_pos=position_ids, mask=masks)
        ref_logits = rlhf.query_response_logits_to_response_logits(ref_logits, context_length)
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self._temperature)

        del ref_logits

        # step 3. estimate values from the generated responses using the value function
        values = self._value_model(query_responses, input_pos=position_ids, mask=masks)
        values = rlhf.query_response_logits_to_response_logits(values, context_length).squeeze(-1)

        # step 4. replace any tokens in the response after the first stop token (usually EOS token) with padding
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
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
        if self._penalise_no_eos or self._min_response_length:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                response_padding_masks,
                seq_lens,
                self._penalise_no_eos,
                self._min_response_length,
            )
            scores[reward_penalty_mask] = self._reward_penalty

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        # step 6.1 values are masked out *after* the last valid token in the response
        value_seq_idxs = torch.where(
            (seq_lens > 0) & (seq_lens < self._max_generated_tokens - 1),
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
        """
        Generates a `self.batch_size` batch of trajectories using `self._forward_batch_size` batch sizes.
        See `generate_trajectory` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory (NamedTuple): A collection of tensors comprising the current trajectory.
        """
        trajectories: List[Trajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[batch_start : batch_start + self._forward_batch_size]
                trajectories.append(self.generate_trajectory(batch_input_ids))
        return Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop."""

        training_completed = False
        pbar = tqdm(total=self._total_steps, initial=self._steps_run)
        for curr_epoch in range(self._epochs_run, self._total_epochs):
            log.info(f"Starting epoch {curr_epoch + 1} of {self._total_epochs}")
            self._sampler.set_epoch(curr_epoch)
            for batchid, batch in enumerate(self._dataloader):
                # if curr_epoch == 1:
                print(f"batch {batchid}")

                # if self._steps_run == 2:

                #

                batch = batch.to(self._device)
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
                    self._kl_coeff,
                    trajectory.value_seq_idxs,
                )

                # step 3. estimate the advantages using Generalized Advantage Estimation (GAE)
                advantages, returns = rlhf.estimate_advantages(
                    trajectory.values,
                    rewards,
                    self._gamma,
                    self._lmbda,
                    masks=~trajectory.response_padding_masks,
                )
                advantages[trajectory.response_padding_masks] = 0.0

                # step 4. optimise using the PPO objective over multiple epochs
                ppo_stats = [
                    [],
                ] * 6
                for _ in range(self._ppo_epochs):
                    batch_idxs = torch.randperm(self.batch_size, device=self._device)
                    for i in range(0, self.batch_size, self._ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self._ppo_batch_size]

                        batch_ppo_stats = [0] * 6
                        for j in range(0, self._ppo_batch_size, self._ppo_backward_batch_size):
                            backward_batch_idxs = mini_batch_idxs[j : j + self._ppo_backward_batch_size]
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
                            print(f"backward step {j}")
                            backward_ppo_stats = self._ppo_step(
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
                        if not self._optimizer_in_bwd:
                            self._optimizer.step()
                            self._optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1

                # step 5. profit
                print(f"completed step: {self._steps_run}")
                self._steps_run += 1
                self.log_metrics(
                    trajectory,
                    kl,
                    kl_rewards,
                    *map(torch.tensor, ppo_stats),
                )
                self.cleanup_after_step(trajectory, advantages, returns, kl, kl_rewards)
                pbar.update(1)
                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            # save checkpoint at current epoch
            self._epochs_run += 1
            # self.save_checkpoint(
            #     curr_epoch, is_intermediate_checkpoint=not training_completed
            # )
            if training_completed:
                break

    def _ppo_step(
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
            pi_logits, trajectory.query_responses[:, context_length:], self._temperature
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0

        # estimate the values from the value function at the current optimisation step
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
        if self._device.type == "cuda" and self._log_peak_memory_stats:
            log_dict.update(utils.get_memory_stats(device=self._device))
        print(log_dict)
        self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup_after_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        """
        Cleanup tensors after each PPO step to free up memory.
        """
        # there shouldn't be any floating references to the individual tensors at the this point, so gc can do its thing
        for v in trajectory:
            del v
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

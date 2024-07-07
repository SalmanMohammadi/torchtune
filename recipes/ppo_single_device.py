# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import sys

from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pkg_resources import packaging
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.datasets import ConcatDataset
from torchtune.models.mistral.modules.transformer import TransformerActorCritic
from torchtune.modules import rlhf
from torchtune.modules.peft.peft_utils import (
    disable_adapter,
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm

log = utils.get_logger("DEBUG")
INVALID_LOGPROBS = 1.0


class PPORecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if self._dtype == torch.bfloat16 and self._device != torch.device("cpu"):
            if torch.cuda.is_available():
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "Full bf16 training is not supported on this hardware."
                    )
            elif torch.backends.mps.is_available():
                if packaging.version.parse(torch.__version__).release < (2, 3):
                    raise RuntimeError(
                        "Full bf16 training is not supported on this hardware."
                    )
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
        - Value head weights with key VALUE_HEAD_KEY
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
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # split off the value head weights
        value_head_state_dict = {
            "value_head.weight": state_dict.pop("value_head.weight"),
            "value_head.bias": state_dict.pop("value_head.bias"),
        }
        ckpt_dict.update({utils.VALUE_HEAD_KEY: value_head_state_dict})

        # save base model as usual
        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})

        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
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
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        ):
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
        model_checkpoint_dict = self._policy_checkpointer.load_checkpoint()

        # load reward model checkpoint
        reward_model_checkpoint_dict = self._reward_checkpointer.load_checkpoint()

        # update recipe state
        if self._resume_from_checkpoint:
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            if utils.ADAPTER_KEY not in model_checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            if utils.VALUE_HEAD_KEY not in model_checkpoint_dict:
                raise ValueError(
                    "Value head weights not found. Please ensure a valid value head checkpoint is provided."
                )
            self._update_recipe_state(model_checkpoint_dict)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model, self._value_model, self._reward_model = self._setup_model(
            cfg_model=cfg.policy,
            cfg_reward_model=cfg.reward_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            model_state_dict=model_checkpoint_dict[utils.MODEL_KEY],
            reward_model_state_dict=reward_model_checkpoint_dict[utils.MODEL_KEY],
            initialise_value_head_from_reward_model=cfg.initialise_value_head_from_reward_model,
        )

        # setup tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # setup opt
        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                model_checkpoint_dict[utils.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
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
                f"exactly divisible by gradient_accumulation_steps ({cfg.gradient_accumulation_steps})."
            )
        self.ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )

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
        self._steps_per_epoch = (
            self.num_steps
        )  # len(self._dataloader) // self.batch_size
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
        value_head_state_dict: Optional[Dict[str, Any]] = None,
        initialise_value_head_from_reward_model: bool = False,
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:

        with utils.set_default_dtype(self._dtype), self._device:
            actor = config.instantiate(cfg_model)
            # exit()
            self._ref_model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_model)
            # if self.share_policy_and_value_backbone:
            #    # this should be fine as long as we don't do anything but
            #    # value_model()
            #     value_model = lambda x: model(x)[1]
            critic = config.instantiate(cfg_reward_model)

        # self._lora_rank = cfg_model.lora_rank
        # self._lora_alpha = cfg_model.lora_alpha
        # self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        # self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        # self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        # self.adapter_params = get_adapter_params(model)
        # set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                actor, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
            utils.set_activation_checkpointing(
                critic, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        # load checkpoints
        # TODO (SalmanMohammadi): remove
        reward_model_state_dict.pop("output.bias")

        reward_model.load_state_dict(reward_model_state_dict)
        critic.load_state_dict(reward_model_state_dict)
        actor.load_state_dict(model_state_dict)
        self._ref_model.load_state_dict(model_state_dict)

        # if model_lora_weights_state_dict:
        #     lora_missing, lora_unexpected = model.load_state_dict(model_lora_weights_state_dict, strict=False)
        # else:
        #     lora_missing, lora_unexpected = None, None

        # value_head_missing, value_head_unexpected = [], []
        # base_missing = [k for k in base_missing if "value_head" not in k]
        # if value_head_state_dict is not None:
        #     value_head_missing, value_head_unexpected = model.load_state_dict(value_head_state_dict, strict=False)
        # else:
        #     if initialise_value_head_from_reward_model:
        #         value_head_state_dict = {}
        #         # reward state dict should be well-formed by this point
        #         # the only error case is mismatched shapes for classification layers
        #         value_head_state_dict["value_head.weight"] = reward_model_state_dict["output.weight"].clone()
        #         if "output.bias" in reward_model_state_dict:
        #             value_head_state_dict["value_head.bias"] = reward_model_state_dict["output.bias"].clone()
        #         try:
        #             model.load_state_dict(value_head_state_dict, strict=False)
        #         except RuntimeError as e:
        #             log.info(
        #                 f"Error loading value head state dict from reward model: {e}"
        #                 "Hint: value head initialisation from reward model is only supported"
        #                 "for reward single-class reward classification models."
        #                 "Reward models which have been trained with bias=True classification layers"
        #                 "require `use_bias`=True in the base model and reward model config."
        #             )
        #             raise
        #         log.info(f"Value model is initialized from reward head")

        # base_missing += value_head_missing
        # base_unexpected += value_head_unexpected

        # validate_missing_and_unexpected_for_lora(
        #     lora_attn_modules=self._lora_attn_modules,
        #     apply_lora_to_mlp=self._apply_lora_to_mlp,
        #     apply_lora_to_output=self._apply_lora_to_output,
        #     base_missing=base_missing,
        #     base_unexpected=base_unexpected,
        #     lora_missing=lora_missing,
        #     lora_unexpected=lora_unexpected,
        # )

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(actor.named_parameters(), dtype=self._dtype)
        log.info(f"Base model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(
            reward_model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Reward model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(
            critic.named_parameters(), dtype=self._dtype
        )
        log.info(f"Value model is initialized with precision {self._dtype}.")

        utils.validate_expected_param_dtype(
            self._ref_model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Ref model is initialized with precision {self._dtype}.")

        for model in [actor, critic, self._ref_model, reward_model]:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    print("Dropout found!!!")
                    assert False
                    module.p = 0

        # parameter_names = [n for n, _ in self._ref_model.named_parameters()]
        # # self._ref_model = copy.deepcopy(model)
        # for param_name in parameter_names:
        #     param = self._ref_model.get_parameter(param_name)
        #     param.requires_grad = False
        # self._ref_model.eval()

        # parameter_names = [n for n, _ in reward_model.named_parameters()]
        # # self._ref_model = copy.deepcopy(model)
        # for param_name in parameter_names:
        #     param = reward_model.get_parameter(param_name)
        #     param.requires_grad = False
        # reward_model.eval()

        # TODO (SalmanMohammadi)
        # Compile model, if enabled.
        # if compile_model:
        #     log.info("Compiling models with torch.compile...")
        #     backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        #     model.compile(backend=backend)

        #     log.info("Compiling reward model with torch.compile...")
        #     reward_model.compile(backend=backend)
        #     if not share_model_value_backbone:
        # log.info("Compiling value model with torch.compile...")
        #     value_model.compile(backend=backend)
        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        # model = TransformerActorCritic(actor=actor, critic=critic)
        reward_model.eval()
        self._ref_model.eval()
        return actor, critic, reward_model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:

        # not sure if chain is necessary?
        optimizer = config.instantiate(
            cfg_optimizer,
            chain(self._model.parameters(), self._value_model.parameters()),
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

    def batched_generate(
        self, input_ids: torch.Tensor, model: nn.Module
    ) -> List[torch.Tensor]:
        """
        Generates sequences using a language model.
        Args:
            input_ids (torch.Tensor): The input tensor of shape [b, seq_len].
            model (nn.Module): The model to generate sequences from.
        Returns:
            torch.Tensor: The concatenated queries and generated sequences of shape
                [b, seq_len + max_generated_tokens].
        """
        outputs = []
        logits = []
        forward_batch_size = min(len(input_ids), self.forward_batch_size)
        for batch_start in range(0, input_ids.shape[0], forward_batch_size):
            batch_end = min(batch_start + forward_batch_size, input_ids.shape[0])
            batch_input_ids = input_ids[batch_start:batch_end]

            # outputs.extend(
            outs, logit = rlhf.generate_with_logits(
                model=model,
                prompt=batch_input_ids,
                max_generated_tokens=self.max_generated_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                # disabling truncation since we require some additional logic to handle truncation
                stop_tokens=None,
                pad_id=self._tokenizer.pad_id,
            )
            outputs.extend(outs)
            logits.extend(logit)
            # )
        return outputs, logits

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
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

        def repeat_generator(dataloader):
            while True:
                yield from dataloader

        log.info("Dataset and Sampler are initialized.")

        return sampler, iter(repeat_generator(dataloader))

    def train(self) -> None:
        """
        The core training loop."""
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            with self._profiler:

                pbar = tqdm(total=self.total_epochs)
                batch = next(iter(self._dataloader))
                if self._profiler_enabled:
                    self._profiler.step()

                # generating the current trajectory in inference mode
                with torch.no_grad():
                    input_ids = batch.to(self._device)
                    batch_size, context_length = input_ids.shape
                    # input ids are left-padded by default - we need to shift the position ids
                    # and generate causal masks if any sequence in the batch has been padded
                    # both during generation and logit calculation
                    # see https://github.com/huggingface/trl/blob/
                    #       f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/utils.py#L1099
                    query_responses, logits = self.batched_generate(
                        input_ids, self._model
                    )
                    query_responses = torch.stack(query_responses)
                    responses = query_responses[:, context_length:].clone()
                    logits = torch.stack(logits)

                    # print(f"gen_logits shape: {logits.shape}")
                    # print(query_responses.shape)
                    # print(f"---- post generationx ---")
                    # print(f"{self._tokenizer.decode(input_ids[0].tolist())}")
                    # print("response ---    ")
                    # print(f"{self._tokenizer.decode(query_responses[0].tolist())}")
                    # print(f"query responses[0]: {query_responses[0].shape}")
                    # create position IDs and causal masks for the current trajectory
                    padding_masks = query_responses == self._tokenizer.pad_id
                    # we only need custom causal masks for sequences with left-padding
                    if padding_masks.any():
                        masks = rlhf.get_causal_mask(~(padding_masks))
                        position_ids = (~padding_masks).cumsum(-1) - (
                            ~padding_masks
                        ).long()
                        position_ids = position_ids.type(torch.int)
                    else:
                        # defer SDPA to handle causal masks
                        masks, position_ids = None, None

                    # print(f"POS ID: {position_ids[0]}")
                    # we only need padding masks for responses from here on

                    # policy and value estimates for the current trajectory
                    # TODO (SalmanMohammadi) implement minibatch model forward pass
                    values = self._value_model(
                        query_responses, input_pos=position_ids, mask=masks
                    )
                    # forward_logits = self._model(query_responses, input_pos=position_ids, mask=masks)
                    # forward_logits_2 = self._model(query_responses, input_pos=position_ids, mask=masks)
                    # print(f"eq double pass: {(forward_logits_2 == forward_logits).all()}")
                    # print(f" logits shape: {logits.shape}")
                    # print(f"logits mean: {forward_logits[:, :-1].mean()}, forward logits m?ean: {logits.mean()}")
                    # num_eq = (forward_logits[:, :-1] == logits).sum()
                    # print(f"all eq: {(forward_logits[:, :-1] == logits).all()}")
                    # # get percentage of elements which are equal
                    # # new = num_eq / (logits[:, :-1].shape[0] * logits[:, :-1].shape[1] * logits[:, :-1].shape[2])
                    # print(
                    #     f"logits vs gen logits: {num_eq / (forward_logits[:, :-1].shape[0] * forward_logits[:, :-1].shape[1] * forward_logits[:, :-1].shape[2])}"
                    # )

                    # print(f"values shape: {values.shape}")
                    # print(f"logits == gen logits")
                    # exit()
                    # logits = logits[:, context_length - 1 :-1]  # [b, max_generated_tokens, vocab_size]
                    # logits = logits[:, context_length - 1 :-1]  # [b, max_generated_tokens, vocab_size]
                    logits = logits[
                        :, context_length - 1 :
                    ]  # [b, max_generated_tokens, vocab_size]
                    logits /= self.temperature
                    # we only need the logprobs of the generated tokens since these are just used for KL rewards
                    logprobs = torch.gather(
                        F.log_softmax(logits, dim=-1),
                        2,
                        responses.unsqueeze(-1),  # [b, max_generated_tokens, 1]
                    ).squeeze(-1)
                    # forward_logprobs = torch.gather(
                    #     F.log_softmax(forward_logits[:, context_length - 1 : -1] / self.temperature, dim=-1),
                    #     2,
                    #     query_responses[:, context_length:].unsqueeze(-1),
                    # ).squeeze(-1)
                    del logits

                    # print(f"logprobs == forward logprobs: {(logprobs == forward_logprobs).all()}")
                    # print(
                    #     f"% of equal elements: {(logprobs == forward_logprobs).sum() / (logprobs.shape[0] * logprobs.shape[1])}"
                    # )
                    ref_logits = self._ref_model(
                        query_responses, input_pos=position_ids, mask=masks
                    )

                    ref_logits = ref_logits[:, context_length - 1 : -1]
                    ref_logits /= self.temperature
                    # shape [b, max_generated_tokens]
                    ref_logprobs = torch.gather(
                        F.log_softmax(ref_logits, dim=-1),
                        2,
                        responses.unsqueeze(-1),
                    ).squeeze(-1)

                    # print(
                    #     f"Logprobs mean: {logprobs.mean()}, ref logprobs mean: {ref_logprobs.mean()}, forward_logprobs mean: {forward_logprobs.mean()}"
                    # )
                    del ref_logits

                    # dear reviewer - would you rather see this as a separate function and tested?
                    # i'm balancing between comprehensability of algo. design decisions and testability here

                    # truncate sequences at the first occurence of eos_id and pad to length
                    (
                        padding_masks,
                        responses,
                    ) = rlhf.truncate_sequence_at_first_stop_token(
                        responses, self.stop_token_ids, self._tokenizer.pad_id
                    )
                    # padding_masks = responses == self._tokenizer.pad_id
                    # eos_mask = torch.isin(query_responses[:, context_length:], self.stop_token_ids)
                    # truncated_responses = torch.where(
                    #     torch.logical_xor(eos_mask.cumsum(-1), eos_mask),
                    #     self._tokenizer.pad_id,
                    #     query_responses[:, context_length:],
                    # )
                    # query_responses[:, context_length:] = torch.where(
                    #     torch.logical_xor(eos_mask.cumsum(-1), eos_mask),
                    #     self._tokenizer.pad_id,
                    #     query_responses[:, context_length:],
                    # )
                    # print(f"query response ids --- ")
                    # print(query_responses[0, context_length:])
                    # print(f"eos_mask ----")
                    # print(eos_mask[0])
                    # print("response after truncate ---    ")
                    # print(f"{self._tokenizer.decode(query_responses[0].tolist())}")

                    # run reward model on truncated query-response sequences: shape [b, context_length + max_generated_tokens]
                    # TODO (SalmanMohammadi): Add support for _reward_model and _model using different tokenizers
                    scores = self._reward_model(
                        torch.cat([input_ids, responses], dim=1),
                        input_pos=position_ids,
                        mask=masks,
                    )
                    # pdb.set_trace()
                    seq_lens = utils.get_unmasked_sequence_lengths(padding_masks)
                    # shape [b, ]
                    scores = scores[
                        torch.arange(0, batch_size), seq_lens + context_length
                    ].squeeze(-1)
                    # scores = torch.ones_like(scores) * 10

                    reward_penalty_mask = torch.zeros_like(scores).to(bool)
                    # to reviewer - see commment above
                    # - sequences without a EOS ID recieve a score of -1
                    if self.penalise_no_eos:
                        reward_penalty_mask = ~padding_masks.any(-1)
                    # - sequences with length < min_response_length recieve a score of -1
                    if self.min_response_length is not None:
                        reward_penalty_mask |= ~(seq_lens >= self.min_response_length)

                    # for seq_idx, eos_mask_ in zip(seq_idxs, padding_masks):
                    #     print(eos_mask_.cumsum(-1).sum(-1) - 1)
                    #     print(seq_idx, ~(eos_mask_.cumsum(-1).sum(-1) - 1 <= self.min_response_length))
                    # see https://arxiv.org/pdf/1909.08593 section 3.1.2
                    scores = torch.where(
                        reward_penalty_mask, self.reward_penalty, scores
                    )
                    # import pdb

                    # pdb.set_trace()
                    # exit()
                    # now mask out logprobs and values w.r.t. padding masks
                    # https://github.com/huggingface/trl/blob/
                    #   f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/ppov2_trainer.py#L359
                    logprobs = torch.where(padding_masks, INVALID_LOGPROBS, logprobs)
                    ref_logprobs = torch.where(
                        padding_masks, INVALID_LOGPROBS, ref_logprobs
                    )

                    # values are masked after the first padding token
                    # see https://github.com/huggingface/trl/blob/f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/
                    #   trl/trainer/ppov2_trainer.py#L354
                    # and the link to the excalidaw therein for a visual explanation
                    value_seq_idxs = torch.where(
                        (seq_lens > 0) & (seq_lens < self.max_generated_tokens - 1),
                        seq_lens + 1,
                        seq_lens,
                    )
                    value_padding_masks = padding_masks.clone()
                    value_padding_masks[
                        torch.arange(batch_size, device=input_ids.device),
                        value_seq_idxs,
                    ] = False
                    values = values[:, context_length - 1 : -1].squeeze(
                        -1
                    )  # [b, max_generated_tokens]

                    values = torch.where(value_padding_masks, 0.0, values)
                    # [b, max_generated_tokens]
                    rewards, kl, kl_rewards = rlhf.get_rewards(
                        scores,
                        logprobs,
                        ref_logprobs,
                        self.kl_controller.value,
                        value_seq_idxs,
                    )
                    if self.whiten_rewards:
                        # shifting mean is disabled for rewards
                        # https://github.com/huggingface/trl/blob/
                        #   f5168fdbaf9cbf6a3f1bdc64dc44b9db3a9ae333/trl/trainer/ppov2_trainer.py#L373
                        rewards = rlhf.masked_whiten(
                            rewards, ~value_padding_masks, shift_mean=False
                        )
                        rewards = torch.where(value_padding_masks, 0.0, rewards)

                    advantages, returns = rlhf.estimate_advantages(
                        values, rewards, self.gamma, self.lmbda, masks=~padding_masks
                    )
                    advantages = torch.where(padding_masks, 0.0, advantages)

                    # useful for tracking
                    num_eos_tokens = (~reward_penalty_mask).sum().item()
                    del seq_lens, value_seq_idxs, responses

                # import pdb

                # pdb.set_trace()
                # trajectory generated! time to optimise
                approx_policy_kls = []
                losses = []
                policy_losses = []
                value_losses = []
                entropies = []
                # print(f"advamages: {advantages[0]}, advantages shape: {advantages.shape}")
                # print(f"returns: {returns[0]} returns shape: {returns.shape}")
                # print(f"values: {values[0]}, values shape: {values.shape}")
                # print("rewards: ", rewards[0], rewards.shape)
                # print(f"query responses: {query_responses.shape}")
                # print(f"--- scores: {scores}")
                # print("response before ppo ---    ")
                # print(f"{self._tokenizer.decode(query_responses[0].tolist())}")
                for _ in range(self.ppo_epochs):
                    # print("--- inside ppo loop ---")
                    # TODO (SalmanMohammadi): Add support for early stopping
                    # shuffle batch indices every epoch
                    batch_idxs = torch.randperm(self.batch_size)
                    # batch_idxs = torch.arange(self.batch_size)
                    # print(batch_idxs)
                    for i in range(0, self.batch_size, self.ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]
                        # print(mini_batch_idxs)
                        running_loss = 0
                        running_policy_loss = 0
                        running_vf_loss = 0
                        for j in range(
                            0, self.ppo_batch_size, self.ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self.ppo_backward_batch_size
                            ]

                            backward_returns = returns[backward_batch_idxs]
                            backward_advantages = advantages[backward_batch_idxs]
                            backward_logprobs = logprobs[backward_batch_idxs]
                            backward_query_responses = query_responses[
                                backward_batch_idxs
                            ]
                            backward_masks = (
                                masks[backward_batch_idxs]
                                if masks is not None
                                else None
                            )
                            backward_position_ids = (
                                position_ids[backward_batch_idxs]
                                if position_ids is not None
                                else None
                            )
                            backward_padding_masks = padding_masks[backward_batch_idxs]
                            backward_value_padding_masks = value_padding_masks[
                                backward_batch_idxs
                            ]
                            backward_values = values[backward_batch_idxs]
                            # print(f"backwards q r shape: {backward_query_responses.shape}")
                            # print(f"logprob = backward logprobs: {logprobs[0] == backward_logprobs}")
                            # policy and value estimates for the current optimisation step
                            # pi_{theta] and V_{phi}
                            # TODO (SalmanMohammadi) implement minibatch model forward pass

                            pi_logits = self._model(
                                backward_query_responses,
                                input_pos=backward_position_ids,
                                mask=backward_masks,
                            )
                            # print(backward_query_responses[0] == query_responses[0])
                            # print(kl[0] == (backward_logprobs[0] - ref_logprobs[0]))
                            # print(f"backwards gen: {self._tokenizer.decode(backward_query_responses[0].tolist())}")
                            # print(f"backwards gen 2 : {self._tokenizer.decode(query_responses[0].tolist())}")
                            # print(f"backwards mask shape {backward_masks.shape}")
                            # print(f"backwards padding mask shape {backward_padding_masks.shape}")
                            # print(f"backwards value mask shape {backward_value_padding_masks.shape}")
                            # print(f"backwards pos id shape {backward_position_ids.shape}")
                            # print(f"backwards pos id  {backward_position_ids[0]}")

                            pi_logits = pi_logits[:, context_length - 1 : -1]
                            pi_logits /= self.temperature
                            pi_logprobs = torch.gather(
                                F.log_softmax(pi_logits, dim=-1),
                                2,
                                backward_query_responses[:, context_length:].unsqueeze(
                                    -1
                                ),
                            ).squeeze(-1)

                            pi_logprobs = torch.where(
                                backward_padding_masks, INVALID_LOGPROBS, pi_logprobs
                            )
                            phi_values = self._value_model(
                                backward_query_responses,
                                input_pos=backward_position_ids,
                                mask=backward_masks,
                            )

                            phi_values = phi_values[:, context_length - 1 : -1].squeeze(
                                -1
                            )
                            phi_values = torch.where(
                                backward_value_padding_masks, 0.0, phi_values
                            )
                            loss, policy_loss, value_loss = self._loss_fn(
                                backward_logprobs,
                                pi_logprobs,
                                backward_advantages,
                                backward_values,
                                phi_values,
                                backward_returns,
                                padding_masks=~backward_padding_masks,
                                value_padding_masks=~backward_value_padding_masks,
                            )
                            import pdb

                            # pdb.set_trace()
                            loss /= self._gradient_accumulation_steps
                            loss.backward()
                            policy_loss /= self._gradient_accumulation_steps
                            value_loss /= self._gradient_accumulation_steps
                            # print(value_loss)
                            # import pdb

                            # pdb.set_trace()
                            running_loss += loss
                            running_policy_loss += policy_loss
                            running_vf_loss += value_loss

                            with torch.no_grad():
                                prob_dist = F.softmax(pi_logits, dim=-1)
                                entropy = torch.logsumexp(
                                    pi_logits, dim=-1
                                ) - torch.sum(prob_dist * pi_logits, dim=-1)
                                entropies.append(entropy.mean())
                                approx_policy_kls.append(
                                    (
                                        0.5 * (pi_logprobs - backward_logprobs).pow(2)
                                    ).mean()
                                )
                                # losses.append(loss.detach().item())
                                # policy_losses.append(policy_loss.item())
                                # value_losses.append(value_loss.item())
                            del (
                                pi_logits,
                                pi_logprobs,
                                backward_logproz_bs,
                                backward_query_responses,
                                backward_masks,
                                backward_position_ids,
                                backward_padding_masks,
                                backward_value_padding_masks,
                                backward_values,
                                backward_returns,
                                backward_advantages,
                            )

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        self.global_step += 1
                        losses.append(running_loss.item())
                        policy_losses.append(running_policy_loss.item())
                        value_losses.append(running_vf_loss.item())

            self.epochs_run += 1
            # self.save_checkpoint(epoch=curr_epoch)
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {curr_epoch+1}| reward: {rewards.sum(1).mean()}"
            )

            kl = logprobs - ref_logprobs
            log_dict = {
                "advantages": torch.tensor(advantages).mean().item(),
                "loss": torch.tensor(losses).mean().item(),
                "mean_per_token_reward": rewards.sum(1).mean().item(),
                "rlhf_reward": scores.mean().item() + kl_rewards.sum(1).mean().item(),
                "kl": kl.sum(1).mean().item(),
                "policy_loss": torch.tensor(policy_losses).mean().item(),
                "value_loss": torch.tensor(value_losses).mean().item(),
                "approx_policy_kl": torch.tensor(approx_policy_kls).mean().item(),
                "entropy": torch.tensor(entropies).mean().item(),
                "num_eos_tokens": num_eos_tokens,
                "scores": scores.mean().item(),
                "values": values.mean().item(),
                "returns": returns.mean().item(),
                "kl_rewards": kl_rewards.sum(1).mean().item(),
            }
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )
            self.kl_controller.update(kl.sum(1).mean().item(), curr_epoch)

            with torch.no_grad():
                toks = torch.tensor(
                    self._tokenizer.encode(
                        "The room was decorated with a", add_eos=False, add_bos=False
                    )
                )
                context_len = len(toks)
                toks = toks.to(self._device).unsqueeze(0)

                outputs, _ = rlhf.generate_with_logits(
                    model=self._model,
                    prompt=toks,
                    max_generated_tokens=50,
                    temperature=self.temperature,
                    top_k=None,
                    stop_tokens=None,
                    pad_id=self._tokenizer.pad_id,
                )

                query_response = self._tokenizer.decode(outputs.squeeze().tolist())
                # columns = ["input", "response"]
                # data = [[query_response[:context_len], query_response[context_len:]]]
                # try:
                #     import wandb
                # import wandb

                # table = wandb.Table(columns=columns, data=data)
                # self._metric_logger.log("query responses", table, step=self.global_step)
                # except ImportError:
                eos_mask, responses = rlhf.truncate_sequence_at_first_stop_token(
                    outputs.clone(), self.stop_token_ids, self._tokenizer.pad_id
                )
                padding_masks = responses == self._tokenizer.pad_id

                seq_lens = utils.get_unmasked_sequence_lengths(padding_masks)
                print(seq_lens)
                print("completion", query_response)
                print(
                    "completion truncated",
                    self._tokenizer.decode(responses.squeeze().tolist()),
                )

            # delete values and clear cache
            del (
                returns,
                advantages,
                logprobs,
                ref_logprobs,
                rewards,
                kl,
                kl_rewards,
                query_responses,
                masks,
                padding_masks,
                value_padding_masks,
                values,
            )
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            elif self._device.type == "mps":
                torch.mps.empty_cache()

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
    config.log_config(recipe_name="PPORecipeSingleDevice", cfg=cfg)
    recipe = PPORecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    torch.save(recipe._model.state_dict(), "model.pt")
    torch.save(recipe._value_model.state_dict(), "value_model.pt")
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

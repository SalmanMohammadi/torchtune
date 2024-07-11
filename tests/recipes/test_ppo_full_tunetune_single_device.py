# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import runpy

import sys
from pathlib import Path

import pytest
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    dummy_text_completion_dataset_config,
    llama2_classifier_test_config,
    llama2_test_config,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
)


class TestPPOFullFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "forward_batch_size=4",
            "ppo_batch_size=4",
            "ppo_epochs=1",
            "num_steps=16",
            "temperature=1.0",
            "grad_accumulation_steps=1",
            "device=cpu",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "seed=9",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ] + dummy_text_completion_dataset_config()

    @pytest.mark.integration_test
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state correctly saved and restored after training."""

        reward_ckpt = "llama2_reward_hf"
        policy_ckpt = "llama2_hf"
        reward_ckpt_path = Path(CKPT_MODEL_PATHS[reward_ckpt])
        policy_ckpt_path = Path(CKPT_MODEL_PATHS[policy_ckpt])

        ckpt_dir = policy_ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        policy_tmpdir = (tmpdir / "policy").mkdir()
        value_tmpdir = (tmpdir / "value").mkdir()
        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(policy_tmpdir)
        write_hf_ckpt_config(value_tmpdir)

        # Train for two steps
        cmd_1 = f"""
        tune run ppo_full_finetune_single_device \
            --config llama2/1B_full_ppo_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{policy_ckpt_path}]\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

            value_checkpointer.checkpoint_dir='{ckpt_dir}' \
            value_checkpointer.checkpoint_files=[{reward_ckpt_path}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            metric_logger._component_=torchtune.utils.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
        """.split()

        model_config = llama2_test_config()
        model_config = [k.replace("model.", "policy_model.") for k in model_config]
        model_config += ["policy_model.intermediate_dim=null"]

        reward_and_value_model_config = llama2_classifier_test_config()
        reward_and_value_model_config = [
            k.replace("model.", "reward_and_value_model.") for k in reward_and_value_model_config
        ]
        reward_and_value_model_config += ["reward_and_value_model.intermediate_dim=null"]
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config + reward_and_value_model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        import pdb

        loss_values = get_loss_values_from_metric_logger(log_file)

        # # Resume training
        resumed_log_dir = (tmpdir / "resumed/").mkdir()
        resumed_log_file = gen_log_file_name(resumed_log_dir)
        print("resuming training ======================")
        cmd_2 = f"""
        tune run ppo_full_finetune_single_device \
            --config llama2/1B_full_ppo_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{policy_tmpdir}' \
            checkpointer.checkpoint_files=[{os.path.join(policy_tmpdir, "hf_model_0001_0.pt")}]\
            checkpointer.recipe_checkpoint={os.path.join(policy_tmpdir, "recipe_state.pt")}\
            checkpointer.output_dir={policy_tmpdir} \
            checkpointer.model_type=LLAMA2 \

            ref_policy_checkpointer.checkpoint_dir='{ckpt_dir}' \
            ref_policy_checkpointer.checkpoint_files=[{policy_ckpt_path}]\

            value_checkpointer.checkpoint_dir='{value_tmpdir}' \
            value_checkpointer.checkpoint_files=[{os.path.join(value_tmpdir, "hf_model_0001_0.pt")}]\
            value_checkpointer.output_dir={value_tmpdir} \

            reward_checkpointer.checkpoint_dir='{ckpt_dir}' \
            reward_checkpointer.checkpoint_files=[{reward_ckpt_path}]\

            resume_from_checkpoint=True \
            metric_logger._component_=torchtune.utils.metric_logging.DiskLogger \
            metric_logger.filename={resumed_log_file} \

        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config + reward_and_value_model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # expected_loss_values = self._fetch_expected_loss_values()[2:]

        new_loss_values = get_loss_values_from_metric_logger(resumed_log_file)

        pdb.set_trace()
        # torch.testing.assert_close(loss_values, expected_loss_values, rtol=1e-4, atol=1e-4)

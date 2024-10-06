# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import textwrap
import json
from pathlib import Path
from typing import Literal, Union
from omegaconf import OmegaConf
from huggingface_hub import create_repo, ModelCardData, ModelCard, whoami, upload_folder, upload_file
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from torchtune._cli.subcommand import Subcommand
import re


def fill_torchtune_model_card_template(
    model_card_data: ModelCardData,
    model_id,
    base_model,
    args,
):

    import importlib, torchao, datasets, sentencepiece

    return f"""
---
{ model_card_data.to_yaml() }
---

# {model_id} 

This model is a finetuned version of [{base_model}](https://huggingface.co/{base_model}) 

# Model description

More information needed

# Training and evaluation results

More information needed

# Training procedure

This model was trained using the [torchtune](https://github.com/pytorch/torchtune) library using the following command:

```bash
{args}
```

# Framework versions

- torchtune {importlib.metadata.version("torchtune")}
- torchao {torchao.__version__}
- datasets {datasets.__version__}
- sentencepiece {sentencepiece.__version__}

"""


class Upload(Subcommand):
    """Holds all the logic for the `tune download` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "upload",
            prog="tune upload",
            usage="tune download <repo-id> [OPTIONS]",
            help="Download a model from the Hugging Face Hub.",
            description="Download a model from the Hugging Face Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the Hugging Face Hub with a Hugging Face API token
                $ tune download meta-llama/Llama-2-7b-hf --hf-token <TOKEN>
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/Llama-2-7b-hf/config.json
                /tmp/Llama-2-7b-hf/README.md
                /tmp/Llama-2-7b-hf/consolidated.00.pth
                ...

                # Download an ungated model from the Hugging Face Hub
                $ tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir /tmp/model
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/model/config.json
                /tmp/model/README.md
                /tmp/model/model-00001-of-00002.bin
                ...

            For a list of all models, visit the Hugging Face Hub https://huggingface.co/models.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._upload_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser."""
        self._parser.add_argument(
            "--repo-id",
            type=str,
            help="Name of the repository on Hugging Face Hub.",
        )
        self._parser.add_argument(
            "--model-id",
            type=str,
            required=False,
            help="Name of the repository on Hugging Face Hub.",
        )
        self._parser.add_argument(
            "--recipe-output-dir",
            type=Path,
            required=True,
            default=None,
        )
        self._parser.add_argument(
            "--tags",
            nargs="+",
            default=[],
            required=False,
            help=(
                "To be used with `output-dir`. If set to 'auto', the cache directory will be used and the file will be"
                " either duplicated or symlinked to the local directory depending on its size. It set to `True`, a"
                " symlink will be created, no matter the file size. If set to `False`, the file will either be"
                " duplicated from cache (if already exists) or downloaded from the Hub and not cached."
            ),
        )
        self._parser.add_argument(
            "--upload-ignore-patterns",
            nargs="+",
            default=[],
            required=False,
            help=(
                "To be used with `output-dir`. If set to 'auto', the cache directory will be used and the file will be"
                " either duplicated or symlinked to the local directory depending on its size. It set to `True`, a"
                " symlink will be created, no matter the file size. If set to `False`, the file will either be"
                " duplicated from cache (if already exists) or downloaded from the Hub and not cached."
            ),
        )
        self._parser.add_argument(
            "--base-model",
            type=str,
            required=False,
            default=None,
        )
        self._parser.add_argument(
            "--datasets",
            nargs="+",
            default=[],
            required=False,
        )
        self._parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2.",
        )

    def _upload_cmd(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Hugging Face Hub."""
        # Download the tokenizer and PyTorch model files

        # Default output_dir is `/tmp/<model_name>`
        # output_dir = args.output_dir
        # if output_dir is None:
        #     model_name = args.repo_id.split("/")[-1]
        #     output_dir = Path("/tmp") / model_name

        # Raise if local_dir_use_symlinks is invalid
        # output_dir_use_symlinks: Union[Literal["auto"], bool]
        # use_symlinks_lowercase = args.output_dir_use_symlinks.lower()
        # if use_symlinks_lowercase == "true":
        #     output_dir_use_symlinks = True
        # elif use_symlinks_lowercase == "false":
        #     output_dir_use_symlinks = False
        # elif use_symlinks_lowercase == "auto":
        #     output_dir_use_symlinks = "auto"
        # else:
        #     self._parser.error(
        #         f"'{args.output_dir_use_symlinks}' is not a valid value for `--output-dir-use-symlinks`. It must be either"
        #         " 'auto', 'True' or 'False'."
        #     )

        meta_path = args.recipe_output_dir / "meta.json"
        config_path = args.recipe_output_dir / "config.yaml"
        config = OmegaConf.load(config_path)

        with open(meta_path) as f:
            recipe_metadata = json.load(f)
        print(recipe_metadata)

        recipe_args = recipe_metadata["sys_argv"]
        new_args = []
        # new_args.append("tune run " + recipe_args[0].split("/")[-1].split(".py")[0])
        new_args.append(" ".join(recipe_args[:3]))
        # new_args.append(" \\\n ".join(recipe_args[3:]))
        # new_args = " \\\n ".join(recipe_args)
        new_args.append(" \\\n ".join(recipe_args[3:]))
        new_args = " \\\n ".join(new_args)
        # print(new_args)
        # exit()
        model_card_data = ModelCardData(
            base_model=args.base_model, datasets=args.datasets, tags=["torchtune"] + args.tags, language="en"
        )
        model_card = ModelCard(
            fill_torchtune_model_card_template(model_card_data, args.model_id, args.base_model, new_args)
        )

        weights_dir = Path(config.checkpointer.output_dir)
        # checkpoint_files = config.checkpointer.
        print(weights_dir.resolve())
        checkpoint_files = config.checkpointer.checkpoint_files
        checkpoint_files = [
            f"hf_model_{int(re.search(r'model-(\d+)-of-\d+\.safetensors', s).group(1)):04d}_0.pt"
            for s in checkpoint_files
        ]
        checkpoint_files = [weights_dir / fname for fname in checkpoint_files]
        hf_checkpoint_files = [str(s).replace(".pt", ".bin") for s in checkpoint_files]
        config_json_path = weights_dir / "config.json"
        tune_adapter_path = weights_dir / "adapter_model.bin"
        peft_adapter_path = weights_dir / "adapter_model.pt"

        from huggingface_hub import whoami, create_repo
        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        user = whoami()["name"]
        repo_id = f"{user}/{args.repo_id}"
        print(repo_id, args.hf_token)
        url = create_repo(repo_id, exist_ok=True, repo_type="model", token=args.hf_token)
        model_card.push_to_hub(repo_id, token=args.hf_token)
        print(checkpoint_files)

        print("Uploading files!")
        # upload_folder(
        #     folder_path=weights_dir,
        #     repo_id=repo_id,
        #     repo_type="model",
        #     ignore_patterns=args.upload_ignore_patterns,
        #     token=args.hf_token,
        # )
        # upload_file(
        #     path_or_fileobj=config_path,
        #     path_in_repo="config.yaml",
        #     repo_id=repo_id,
        #     repo_type="model",
        #     token=args.hf_token,
        # )
        exit()
        # print(f"Ignoring files matching the following patterns: {args.ignore_patterns}")
        try:
            pass
            # true_output_dir = snapshot_download(
            #     args.repo_id,
            #     local_dir=output_dir,
            #     local_dir_use_symlinks=output_dir_use_symlinks,
            #     ignore_patterns=args.ignore_patterns,
            #     token=args.hf_token,
            # )
        except GatedRepoError:
            self._parser.error(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository and have provided the proper Hugging Face API token "
                "using the option `--hf-token` or by running `huggingface-cli login`."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
        except RepositoryNotFoundError:
            self._parser.error(f"Repository '{args.repo_id}' not found on the Hugging Face Hub.")
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            msg = f"Failed to download {args.repo_id} with error: '{e}' and traceback: {tb}"
            self._parser.error(msg)

        print(
            "Successfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )

    def create_model_card(self) -> str:
        pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from functools import lru_cache, wraps
from typing import Callable, Optional, TypeVar
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
import sys
import importlib, torchao
import datasets

T = TypeVar("T", bound=type)


def get_logger(level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (Optional[str]): The logging level. See https://docs.python.org/3/library/logging.html#levels for list of levels.

    Example:
        >>> logger = get_logger("INFO")
        >>> logger.info("Hello world!")
        INFO:torchtune.utils._logging:Hello world!

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


@lru_cache(None)
def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)


def deprecated(msg: str = "") -> Callable[[T], T]:
    """
    Decorator to mark an object as deprecated and print additional message.

    Args:
        msg (str): additional information to print after warning.

    Returns:
        Callable[[T], T]: the decorated object.
    """

    @lru_cache(maxsize=1)
    def warn(obj):
        warnings.warn(
            f"{obj.__name__} is deprecated and will be removed in future versions. " + msg,
            category=FutureWarning,
            stacklevel=3,
        )

    def decorator(obj):
        @wraps(obj)
        def wrapper(*args, **kwargs):
            warn(obj)
            return obj(*args, **kwargs)

        return wrapper

    return decorator


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg)


from pathlib import Path
import json


def write_recipe_artefacts_to_output_dir_rank_zero(cfg: DictConfig):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    OmegaConf.save(cfg, Path(cfg.output_dir) / "config.yaml")
    recipe_metadata = {}
    recipe_metadata["sys_argv"] = sys.argv
    recipe_metadata["torchtune_version"] = importlib.metadata.version("torchtune")
    recipe_metadata["torchao_version"] = torchao.__version__
    recipe_metadata["datasets_version"] = datasets.__version__
    with open(Path(cfg.output_dir) / "meta.json", "w") as f:
        json.dump(recipe_metadata, f)

"""Shared utilities for training scripts."""

import logging
import os
import random
from datetime import date

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set seeds for random, numpy, torch, and torch.cuda (when available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def setup_logging(output_dir: str, log_filename: str = "training.log") -> str:
    """Configure file logging for a training run.

    Creates a dated sub-directory ``<output_dir>/<YYYY-MM-DD>/`` (if it does not
    already exist), attaches a :class:`logging.FileHandler` that writes every log
    record to ``<dated_dir>/<log_filename>``, and returns the path to the dated
    directory.

    Call this function *after* :func:`logging.basicConfig` has been called so that
    the root logger's level and console handler are already in place.

    Args:
        output_dir: Base experiment output directory.
        log_filename: Name of the log file inside the dated sub-directory.

    Returns:
        Path to the dated sub-directory (e.g. ``<output_dir>/2026-03-29``).
    """
    dated_dir = os.path.join(output_dir, date.today().isoformat())
    os.makedirs(dated_dir, exist_ok=True)

    log_path = os.path.join(dated_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    return dated_dir


def get_lambda_lr(epoch: int, n_epochs: int, n_epochs_decay: int) -> float:
    """Linear learning-rate decay schedule used by all CycleGAN variants.

    Returns 1.0 for the first *n_epochs* epochs, then linearly decays to 0
    over the following *n_epochs_decay* epochs.
    """
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / float(n_epochs_decay + 1))

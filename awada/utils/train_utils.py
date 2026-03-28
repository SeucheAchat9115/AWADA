"""Shared utilities for training scripts."""

import random

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


def get_lambda_lr(epoch: int, n_epochs: int, n_epochs_decay: int) -> float:
    """Linear learning-rate decay schedule used by all CycleGAN variants.

    Returns 1.0 for the first *n_epochs* epochs, then linearly decays to 0
    over the following *n_epochs_decay* epochs.
    """
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / float(n_epochs_decay + 1))

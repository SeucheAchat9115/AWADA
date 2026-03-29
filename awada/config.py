"""Centralised configuration constants for AWADA.

All constants are loaded from YAML files under ``configs/datasets/`` so they
can be edited without touching Python source code.  The public names exported
from this module remain unchanged, ensuring full backward compatibility with
all dataset and model files that import from here.
"""

from __future__ import annotations

import pathlib

import torch
import yaml

# Root of the configs/datasets/ directory (relative to this file's package).
_DATASETS_DIR = pathlib.Path(__file__).parent.parent / "configs" / "datasets"


def _load(name: str) -> dict:
    with open(_DATASETS_DIR / name) as f:
        return yaml.safe_load(f)


_cityscapes = _load("cityscapes.yaml")
_bdd100k = _load("bdd100k.yaml")
_sim10k = _load("sim10k.yaml")
_norm = _load("normalization.yaml")

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Default compute device: prefer CUDA when available, fall back to CPU.
DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# ImageNet normalisation (used by the DeepLabV3 semantic-loss backbone)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: list[float] = _norm["mean"]
IMAGENET_STD: list[float] = _norm["std"]

# ---------------------------------------------------------------------------
# Cityscapes dataset constants
# ---------------------------------------------------------------------------

# Cityscapes instance ID → 1-based detection label (8-class, default map).
# Instance IDs follow the convention class_id * 1000 + instance_id.
CITYSCAPES_LABEL_MAP: dict[int, int] = {
    int(k): v for k, v in _cityscapes["label_map"].items()
}

# Human-readable class names corresponding to the 8-class label map above.
CITYSCAPES_CLASS_NAMES: list[str] = _cityscapes["class_names"]

# 7-class label map used for the Cityscapes → BDD100k benchmark.
# The "train" class (Cityscapes ID 31) is absent from BDD100k and is
# excluded from both sides of the benchmark.  Motorcycle and bicycle are
# renumbered so that label IDs are contiguous and match BDD100K_LABEL_MAP.
CITYSCAPES_BDD100K_LABEL_MAP: dict[int, int] = {
    int(k): v for k, v in _cityscapes["bdd100k_label_map"].items()
}

# Human-readable class names for the 7-class BDD100k-aligned Cityscapes map.
BDD100K_ALIGNED_CLASSES: list[str] = _cityscapes["bdd100k_aligned_classes"]

# Minimum number of foreground pixels for an instance to be kept as a box.
MIN_PIXELS_THRESHOLD: int = _cityscapes["min_pixels_threshold"]

# ---------------------------------------------------------------------------
# BDD100k dataset constants
# ---------------------------------------------------------------------------

# BDD100k detection categories aligned with the Cityscapes 7-class benchmark.
# Included: pedestrian, rider, car, truck, bus, motorcycle, bicycle.
BDD100K_LABEL_MAP: dict[str, int] = _bdd100k["label_map"]

# Human-readable class names corresponding to BDD100K_LABEL_MAP.
BDD100K_CLASS_NAMES: list[str] = _bdd100k["class_names"]

# Minimum box dimension (width or height) below which detections are discarded.
MIN_BOX_DIM: int = _bdd100k["min_box_dim"]

# ---------------------------------------------------------------------------
# Sim10k dataset constants
# ---------------------------------------------------------------------------

# Sim10k only annotates the single "car" class.
SIM10K_CLASS_NAMES: list[str] = _sim10k["class_names"]

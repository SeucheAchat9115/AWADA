"""Centralised configuration constants for AWADA.

All module-level globals that were previously scattered across model and
dataset files are collected here so that they are easy to inspect, override
in tests, and keep consistent across the code-base.
"""

import torch

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Default compute device: prefer CUDA when available, fall back to CPU.
DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# ImageNet normalisation (used by the DeepLabV3 semantic-loss backbone)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Cityscapes dataset constants
# ---------------------------------------------------------------------------

# Cityscapes instance ID → 1-based detection label (8-class, default map).
# Instance IDs follow the convention class_id * 1000 + instance_id.
CITYSCAPES_LABEL_MAP: dict[int, int] = {
    24: 1,  # person
    25: 2,  # rider
    26: 3,  # car
    27: 4,  # truck
    28: 5,  # bus
    31: 6,  # train
    32: 7,  # motorcycle
    33: 8,  # bicycle
}

# Human-readable class names corresponding to the 8-class label map above.
CITYSCAPES_CLASS_NAMES: list[str] = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# 7-class label map used for the Cityscapes → BDD100k benchmark.
# The "train" class (Cityscapes ID 31) is absent from BDD100k and is
# excluded from both sides of the benchmark.  Motorcycle and bicycle are
# renumbered so that label IDs are contiguous and match BDD100K_LABEL_MAP.
CITYSCAPES_BDD100K_LABEL_MAP: dict[int, int] = {
    24: 1,  # person
    25: 2,  # rider
    26: 3,  # car
    27: 4,  # truck
    28: 5,  # bus
    # 31 (train) excluded
    32: 6,  # motorcycle
    33: 7,  # bicycle
}

# Human-readable class names for the 7-class BDD100k-aligned Cityscapes map.
BDD100K_ALIGNED_CLASSES: list[str] = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

# Minimum number of foreground pixels for an instance to be kept as a box.
MIN_PIXELS_THRESHOLD: int = 10

# ---------------------------------------------------------------------------
# BDD100k dataset constants
# ---------------------------------------------------------------------------

# BDD100k detection categories aligned with the Cityscapes 7-class benchmark.
# Included: pedestrian, rider, car, truck, bus, motorcycle, bicycle.
BDD100K_LABEL_MAP: dict[str, int] = {
    "pedestrian": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "motorcycle": 6,
    "bicycle": 7,
}

# Human-readable class names corresponding to BDD100K_LABEL_MAP.
BDD100K_CLASS_NAMES: list[str] = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

# Minimum box dimension (width or height) below which detections are discarded.
MIN_BOX_DIM: int = 5

# ---------------------------------------------------------------------------
# Sim10k dataset constants
# ---------------------------------------------------------------------------

# Sim10k only annotates the single "car" class.
SIM10K_CLASS_NAMES: list[str] = ["car"]

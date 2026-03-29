"""Hydra structured configuration dataclasses for AWADA.

These dataclasses define the configuration schema and provide type-safe
defaults for all parameter groups used across training and evaluation scripts.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING


@dataclass
class TrainingConfig:
    """Common training hyperparameters shared by all GAN training scripts."""
    epochs: int = 200
    lr: float = 0.0002
    betas: List[float] = field(default_factory=lambda: [0.5, 0.999])
    batch_size: int = 1
    patch_size: int = 128
    resize_min_side: int = 600
    seed: int = 42
    amp: bool = False
    save_every: int = 10
    resume: Optional[str] = None
    log_interval: int = 100


@dataclass
class ModelConfig:
    """GAN loss weights and replay buffer settings."""
    lambda_gan: float = 1.0
    lambda_cyc: float = 10.0
    lambda_idt: float = 0.0
    lambda_sem: float = 0.0
    buffer_size: int = 50
    buffer_return_prob: float = 0.5
    disc_loss_avg_factor: float = 0.5


@dataclass
class DataConfig:
    """Dataset paths for unpaired image translation (CycleGAN / CyCada)."""
    source_dir: str = MISSING
    target_dir: str = MISSING
    output_dir: str = MISSING


@dataclass
class AwadaDataConfig(DataConfig):
    """Dataset paths for AWADA (extends DataConfig with attention map dirs)."""
    source_attention_dir: str = MISSING
    target_attention_dir: str = MISSING


@dataclass
class HardwareConfig:
    """Hardware / device settings."""
    device: str = "cpu"


@dataclass
class CityscapesDatasetConfig:
    """Cityscapes dataset constants."""
    class_names: List[str] = field(
        default_factory=lambda: [
            "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]
    )
    min_pixels_threshold: int = 10
    bdd100k_aligned_classes: List[str] = field(
        default_factory=lambda: [
            "person", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
        ]
    )


@dataclass
class Bdd100kDatasetConfig:
    """BDD100k dataset constants."""
    class_names: List[str] = field(
        default_factory=lambda: [
            "pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
        ]
    )
    min_box_dim: int = 5


@dataclass
class Sim10kDatasetConfig:
    """Sim10k dataset constants."""
    class_names: List[str] = field(default_factory=lambda: ["car"])


@dataclass
class DetectorConfig:
    """Faster R-CNN detector training / evaluation settings."""
    dataset: str = MISSING
    data_root: str = MISSING
    num_classes: int = MISSING
    output_dir: str = MISSING
    epochs: int = 10
    batch_size: int = 2
    lr: float = 0.005
    device: str = "cpu"
    pretrained: bool = True
    classes: Optional[List[str]] = None
    image_dir: Optional[str] = None
    resize: Optional[int] = None
    seed: int = 42
    log_interval: int = 100
    amp: bool = False


@dataclass
class AttentionMapConfig:
    """Configuration for the generate_attention_maps script."""
    detector_checkpoint: str = MISSING
    dataset: str = MISSING
    data_root: str = MISSING
    output_dir: str = MISSING
    num_classes: int = MISSING
    score_threshold: float = 0.5
    split: str = "train"
    device: str = "cpu"
    seed: int = 42


@dataclass
class EvaluateDetectorConfig:
    """Configuration for the evaluate_detector script."""
    detector_checkpoint: str = MISSING
    dataset: str = MISSING
    data_root: str = MISSING
    num_classes: int = MISSING
    output_dir: str = MISSING
    split: str = "val"
    device: str = "cpu"
    classes: Optional[List[str]] = None
    resize: Optional[int] = None
    label: str = ""
    benchmark: str = ""


@dataclass
class StylizeConfig:
    """Configuration for the stylize_dataset script."""
    generator_checkpoint: str = MISSING
    source_dir: str = MISSING
    output_dir: str = MISSING
    device: str = "cpu"


@dataclass
class VisualizeConfig:
    """Configuration for the visualize_inference script."""
    checkpoint: str = MISSING
    input_dir: str = MISSING
    output_dir: str = MISSING
    direction: str = "AB"
    num_images: Optional[int] = None
    device: str = "cpu"

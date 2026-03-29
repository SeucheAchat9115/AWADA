#!/usr/bin/env python3
"""Generate RPN attention maps from a trained Faster R-CNN detector."""

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from awada.datasets.bdd100k import Bdd100kDetectionDataset
from awada.datasets.cityscapes import CityscapesDetectionDataset
from awada.datasets.foggy_cityscapes import FoggyCityscapesDetectionDataset
from awada.datasets.sim10k import Sim10kDetectionDataset
from awada.utils.attention import generate_attention_maps
from awada.utils.train_utils import set_seed


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _generate(cfg: DictConfig) -> None:
    """Generate attention maps from a Hydra config."""
    set_seed(cfg.attention.seed)

    device = torch.device(cfg.hardware.device)

    if cfg.attention.dataset == "sim10k":
        dataset = Sim10kDetectionDataset(cfg.attention.data_root)
    elif cfg.attention.dataset == "cityscapes":
        dataset = CityscapesDetectionDataset(cfg.attention.data_root, split=cfg.attention.split)
    elif cfg.attention.dataset == "bdd100k":
        dataset = Bdd100kDetectionDataset(cfg.attention.data_root, split=cfg.attention.split)
    else:
        dataset = FoggyCityscapesDetectionDataset(
            cfg.attention.data_root, split=cfg.attention.split
        )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    model = build_model(cfg.attention.num_classes)
    state = torch.load(cfg.attention.detector_checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    generate_attention_maps(
        model,
        dataloader,
        cfg.attention.output_dir,
        score_threshold=cfg.attention.score_threshold,
        device=str(device),
    )


@hydra.main(
    version_base=None, config_path="../configs", config_name="generate_attention_maps"
)
def main(cfg: DictConfig) -> None:
    _generate(cfg)


if __name__ == "__main__":
    main()

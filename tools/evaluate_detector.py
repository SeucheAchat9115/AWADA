#!/usr/bin/env python3
"""Evaluate a trained Faster R-CNN detector on a target-domain dataset.

This script loads a detector checkpoint, runs inference on the validation
split of a target dataset, and reports mAP@0.5 and mAP@0.5:0.95.  Results
are also written to ``<output_dir>/results.txt`` so that shell experiment
scripts can call this instead of embedding inline Python.

Example usage::

    python evaluate_detector.py \\
        evaluate.detector_checkpoint=outputs/exp_a/detector_final.pth \\
        evaluate.dataset=cityscapes \\
        evaluate.data_root=/data/cityscapes \\
        evaluate.num_classes=1 \\
        evaluate.output_dir=outputs/exp_a \\
        evaluate.label="Experiment A: Non-Adaptive Baseline" \\
        evaluate.benchmark=sim10k_to_cityscapes \\
        evaluate.classes=[car] \\
        hardware.device=cuda
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from awada.datasets.bdd100k import Bdd100kDetectionDataset
from awada.datasets.cityscapes import CityscapesDetectionDataset
from awada.datasets.foggy_cityscapes import FoggyCityscapesDetectionDataset
from awada.utils.metrics import compute_map_range
from awada.utils.transforms import ResizeToMinSize

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    """Build a Faster R-CNN ResNet50-FPN model with a custom number of classes."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1
    )  # +1 for background (class 0)
    return model


def get_dataset(name, root, split, classes=None, transforms=None):
    """Return the validation dataset for *name*."""
    if name == "cityscapes":
        return CityscapesDetectionDataset(root, split=split, classes=classes, transforms=transforms)
    elif name == "foggy_cityscapes":
        return FoggyCityscapesDetectionDataset(root, split=split, transforms=transforms)
    elif name == "bdd100k":
        return Bdd100kDetectionDataset(root, split=split, transforms=transforms)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from *checkpoint_path*.

    Handles both plain ``state_dict`` files (saved with
    ``torch.save(model.state_dict(), path)``) and checkpoint dicts that
    contain a ``"model_state_dict"`` key (saved during training).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def evaluate(model, dataloader, device, num_classes):
    """Run inference on *dataloader* and return mAP metrics."""
    model.eval()
    predictions, targets_all = [], []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out in outputs:
                predictions.append({k: v.cpu() for k, v in out.items()})
            for t in targets:
                targets_all.append(
                    {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                )
    return compute_map_range(predictions, targets_all, num_classes=num_classes)


def _evaluate(cfg: DictConfig) -> None:
    """Run detector evaluation from a Hydra config."""
    device = torch.device(cfg.hardware.device)

    classes = list(cfg.evaluate.classes) if cfg.evaluate.classes is not None else None
    resize_transform = (
        ResizeToMinSize(cfg.evaluate.resize) if cfg.evaluate.resize is not None else None
    )
    dataset = get_dataset(
        cfg.evaluate.dataset,
        cfg.evaluate.data_root,
        cfg.evaluate.split,
        classes=classes,
        transforms=resize_transform,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(cfg.evaluate.num_classes)
    model = load_checkpoint(model, cfg.evaluate.detector_checkpoint, device)
    model.to(device)

    metrics = evaluate(model, loader, device, cfg.evaluate.num_classes)

    label = cfg.evaluate.label or f"Detector on {cfg.evaluate.dataset}"
    print(f"{label}:")
    print(f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")
    per_class_AP = metrics.get("per_class_AP", {})
    if per_class_AP:
        print("  Per-class AP@0.5:0.95:")
        for cat_id, ap in sorted(per_class_AP.items()):
            print(f"    class {cat_id}: {ap:.4f}")

    os.makedirs(cfg.evaluate.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.evaluate.output_dir, "results.txt")
    with open(results_path, "w") as f:
        if cfg.evaluate.label:
            f.write(f"{cfg.evaluate.label}\n")
        if cfg.evaluate.benchmark:
            f.write(f"Benchmark: {cfg.evaluate.benchmark}\n")
        f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
    print(f"Results saved to {results_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_detector")
def main(cfg: DictConfig) -> None:
    _evaluate(cfg)


if __name__ == "__main__":
    main()

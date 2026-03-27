#!/usr/bin/env python3
"""Evaluate a trained Faster R-CNN detector on a target-domain dataset.

This script loads a detector checkpoint, runs inference on the validation
split of a target dataset, and reports mAP@0.5 and mAP@0.5:0.95.  Results
are also written to ``<output_dir>/results.txt`` so that shell experiment
scripts can call this instead of embedding inline Python.

Example usage::

    python evaluate_detector.py \\
        --detector_checkpoint outputs/exp_a/detector_final.pth \\
        --dataset cityscapes \\
        --data_root /data/cityscapes \\
        --num_classes 1 \\
        --output_dir outputs/exp_a \\
        --label "Experiment A: Non-Adaptive Baseline" \\
        --benchmark sim10k_to_cityscapes \\
        --classes car \\
        --device cuda
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.datasets.bdd100k import Bdd100kDataset
from src.datasets.cityscapes import CityscapesDetectionDataset
from src.datasets.foggy_cityscapes import FoggyCityscapesDataset
from src.utils.metrics import compute_map_range
from src.utils.transforms import ResizeToMinSize


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
        return FoggyCityscapesDataset(root, split=split, transforms=transforms)
    elif name == "bdd100k":
        return Bdd100kDataset(root, split=split, transforms=transforms)
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Faster R-CNN detector on a target-domain dataset"
    )
    parser.add_argument(
        "--detector_checkpoint",
        required=True,
        help="Path to the detector checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset",
        choices=["cityscapes", "foggy_cityscapes", "bdd100k"],
        required=True,
        help="Target dataset to evaluate on",
    )
    parser.add_argument("--data_root", required=True, help="Root directory of the dataset")
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of foreground classes"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where results.txt will be written"
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names to include (cityscapes only, e.g. --classes car)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        metavar="MIN_SIZE",
        help=(
            "Resize images so the shortest side equals MIN_SIZE pixels before evaluation "
            "(e.g. --resize 600).  Also scales bounding boxes accordingly.  "
            "Works with both original and stylized images."
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help="Human-readable experiment label written to results.txt (e.g. 'Experiment A: ...')",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="Benchmark identifier written to results.txt (e.g. sim10k_to_cityscapes)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    resize_transform = ResizeToMinSize(args.resize) if args.resize is not None else None
    dataset = get_dataset(
        args.dataset, args.data_root, args.split, classes=args.classes, transforms=resize_transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(args.num_classes)
    model = load_checkpoint(model, args.detector_checkpoint, device)
    model.to(device)

    metrics = evaluate(model, loader, device, args.num_classes)

    label = args.label or f"Detector on {args.dataset}"
    print(f"{label}:")
    print(f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, "w") as f:
        if args.label:
            f.write(f"{args.label}\n")
        if args.benchmark:
            f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

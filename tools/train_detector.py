#!/usr/bin/env python3
"""Train Faster R-CNN on a source or target domain dataset."""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from awada.datasets.bdd100k import Bdd100kDetectionDataset
from awada.datasets.cityscapes import CityscapesDetectionDataset
from awada.datasets.foggy_cityscapes import FoggyCityscapesDetectionDataset
from awada.datasets.sim10k import Sim10kDetectionDataset
from awada.utils.metrics import compute_map_range
from awada.utils.train_utils import set_seed, setup_logging
from awada.utils.transforms import ResizeToMinSize

logger = logging.getLogger(__name__)


def get_dataset(name, root, split, transforms=None, classes=None, image_dir=None):
    if name == "sim10k":
        return Sim10kDetectionDataset(root, transforms=transforms, image_dir=image_dir)
    elif name == "cityscapes":
        return CityscapesDetectionDataset(
            root, split=split, transforms=transforms, classes=classes, image_root=image_dir
        )
    elif name == "foggy_cityscapes":
        return FoggyCityscapesDetectionDataset(
            root, split=split, transforms=transforms, image_root=image_dir
        )
    elif name == "bdd100k":
        return Bdd100kDetectionDataset(
            root, split=split, transforms=transforms, image_root=image_dir
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1
    )  # +1 to include background (class 0)
    return model


def evaluate(model, dataloader, device, num_classes):
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train Faster R-CNN detector")
    parser.add_argument(
        "--dataset", choices=["sim10k", "cityscapes", "foggy_cityscapes", "bdd100k"], required=True
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names to include (cityscapes only, e.g. --classes car)",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help=(
            "Override the default image directory with a custom path. "
            "Useful for training on stylized images while keeping the original annotations. "
            "For sim10k: path to a flat directory of image files. "
            "For cityscapes/foggy_cityscapes: path to a directory with the same city-based "
            "subdirectory structure as the standard image root."
        ),
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        metavar="MIN_SIZE",
        help=(
            "Resize images so the shortest side equals MIN_SIZE pixels before training "
            "(e.g. --resize 600).  Also scales bounding boxes accordingly.  "
            "Works with both original and stylized images."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log training loss every N iterations",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable Automatic Mixed Precision (AMP) training",
    )
    parser.add_argument(
        "--val_dataset",
        choices=["sim10k", "cityscapes", "foggy_cityscapes", "bdd100k"],
        default=None,
        help=(
            "Dataset to use for validation (detection evaluation). "
            "In a domain adaptation setting, set this to the target dataset. "
            "Defaults to the training dataset if not specified."
        ),
    )
    parser.add_argument(
        "--val_data_root",
        default=None,
        help=(
            "Root directory of the validation dataset. "
            "Required when --val_dataset is different from --dataset."
        ),
    )

    args = parser.parse_args()

    if (
        args.val_dataset is not None
        and args.val_dataset != args.dataset
        and args.val_data_root is None
    ):
        parser.error("--val_data_root is required when --val_dataset differs from --dataset")

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    device = torch.device(args.device)

    resize_transform = ResizeToMinSize(args.resize) if args.resize is not None else None
    train_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split="train",
        classes=args.classes,
        image_dir=args.image_dir,
        transforms=resize_transform,
    )
    # In a domain adaptation setting, validate on the target dataset when specified.
    val_dataset_name = args.val_dataset if args.val_dataset is not None else args.dataset
    val_data_root = args.val_data_root if args.val_data_root is not None else args.data_root
    val_dataset = get_dataset(
        val_dataset_name,
        val_data_root,
        split="val",
        classes=args.classes,
        transforms=resize_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
        if val_dataset is not None
        else None
    )

    model = build_model(args.num_classes, pretrained=args.pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    final_metrics = None
    for epoch in range(args.epochs):
        logger.info("--- Epoch %d/%d ---", epoch + 1, args.epochs)
        model.train()
        running_loss = 0.0
        epoch_total_loss = 0.0
        epoch_iters = 0
        for iteration, (images, targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            # Filter out images with no annotations
            valid = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
            if not valid:
                continue
            images, targets = zip(*valid)

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss_dict = model(list(images), list(targets))
                losses = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()
            epoch_total_loss += losses.item()
            epoch_iters += 1
            if (iteration + 1) % args.log_interval == 0:
                logger.info(
                    "  [Epoch %d, Iter %d] Loss: %.4f",
                    epoch + 1,
                    iteration + 1,
                    running_loss / args.log_interval,
                )
                running_loss = 0.0

        logger.info(
            "Epoch %d complete | avg total loss=%.4f",
            epoch + 1,
            epoch_total_loss / max(epoch_iters, 1),
        )

        scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"detector_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved: %s", ckpt_path)

        # Evaluate
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args.num_classes)
            logger.info(
                "Epoch %d Validation: mAP@0.5=%.4f, mAP@0.5:0.95=%.4f",
                epoch + 1,
                metrics["mAP@0.5"],
                metrics["mAP@0.5:0.95"],
            )
            for cat_id, ap in sorted(metrics["per_class_AP"].items()):
                logger.info("  Class %d AP@0.5:0.95=%.4f", cat_id, ap)
            final_metrics = metrics

    # Save final model
    final_path = os.path.join(args.output_dir, "detector_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    # Write final validation metrics to results.txt
    if val_loader is not None and final_metrics is not None:
        results_path = os.path.join(args.output_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write(f"mAP@0.5: {final_metrics['mAP@0.5']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {final_metrics['mAP@0.5:0.95']:.4f}\n")
        logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()

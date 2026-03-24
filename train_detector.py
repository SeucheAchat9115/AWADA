#!/usr/bin/env python3
"""Train Faster R-CNN on a source or target domain dataset."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.datasets.sim10k import Sim10kDataset
from src.datasets.cityscapes import CityscapesDetectionDataset
from src.datasets.foggy_cityscapes import FoggyCityscapesDataset
from src.utils.metrics import compute_map_range


def get_dataset(name, root, split, transforms=None, classes=None):
    if name == 'sim10k':
        return Sim10kDataset(root, transforms=transforms)
    elif name == 'cityscapes':
        return CityscapesDetectionDataset(root, split=split, transforms=transforms, classes=classes)
    elif name == 'foggy_cityscapes':
        return FoggyCityscapesDataset(root, split=split, transforms=transforms)
    else:
        raise ValueError(f'Unknown dataset: {name}')


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 to include background (class 0)
    return model


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    predictions, targets_all = [], []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out in outputs:
                predictions.append({k: v.cpu() for k, v in out.items()})
            for t in targets:
                targets_all.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()})
    return compute_map_range(predictions, targets_all, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN detector')
    parser.add_argument('--dataset', choices=['sim10k', 'cityscapes', 'foggy_cityscapes'], required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='Class names to include (cityscapes only, e.g. --classes car)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = get_dataset(args.dataset, args.data_root, split='train', classes=args.classes)
    # sim10k (GTA) does not require a validation split; all images are used for training.
    val_dataset = get_dataset(args.dataset, args.data_root, split='val', classes=args.classes) if args.dataset != 'sim10k' else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=collate_fn) if val_dataset is not None else None

    model = build_model(args.num_classes, pretrained=args.pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for iteration, (images, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                       for t in targets]

            # Filter out images with no annotations
            valid = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            if not valid:
                continue
            images, targets = zip(*valid)

            loss_dict = model(list(images), list(targets))
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if (iteration + 1) % 100 == 0:
                print(f'  [Epoch {epoch+1}, Iter {iteration+1}] Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f'detector_epoch_{epoch+1}.pth')
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

        # Evaluate
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args.num_classes)
            print(f'Epoch {epoch+1} Validation: mAP@0.5={metrics["mAP@0.5"]:.4f}, mAP@0.5:0.95={metrics["mAP@0.5:0.95"]:.4f}')

    # Save final model
    final_path = os.path.join(args.output_dir, 'detector_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')


if __name__ == '__main__':
    main()

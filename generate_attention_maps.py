#!/usr/bin/env python3
"""Generate RPN attention maps from a trained Faster R-CNN detector."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.datasets.gta5 import GTA5Dataset
from src.datasets.cityscapes import CityscapesDetectionDataset
from src.utils.attention import generate_attention_maps


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate RPN attention maps')
    parser.add_argument('--detector_checkpoint', required=True)
    parser.add_argument('--dataset', choices=['gta5', 'cityscapes'], required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dataset == 'gta5':
        dataset = GTA5Dataset(args.data_root, split=args.split)
    else:
        dataset = CityscapesDetectionDataset(args.data_root, split=args.split)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    model = build_model(args.num_classes)
    state = torch.load(args.detector_checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    generate_attention_maps(model, dataloader, args.output_dir,
                            top_k=args.top_k, device=str(device))


if __name__ == '__main__':
    main()

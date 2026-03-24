import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .cityscapes import CITYSCAPES_LABEL_MAP, MIN_PIXELS_THRESHOLD


class FoggyCityscapesDataset(Dataset):
    def __init__(self, root, split='val', beta=0.02, transforms=None):
        self.root = root
        self.split = split
        self.beta = beta
        self.transforms = transforms
        self.samples = []

        img_base = os.path.join(root, 'leftImg8bit_foggy', split)
        ann_base = os.path.join(root, 'gtFine', split)

        for city in sorted(os.listdir(img_base)):
            city_img_dir = os.path.join(img_base, city)
            city_ann_dir = os.path.join(ann_base, city)
            if not os.path.isdir(city_img_dir):
                continue
            for fname in sorted(os.listdir(city_img_dir)):
                suffix = f'_leftImg8bit_foggy_beta_{beta:.2f}.png'
                if not fname.endswith(suffix):
                    continue
                stem = fname.replace(suffix, '')
                ann_fname = stem + '_gtFine_instanceIds.png'
                ann_path = os.path.join(city_ann_dir, ann_fname)
                if os.path.exists(ann_path):
                    self.samples.append((os.path.join(city_img_dir, fname), ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        instance_map = np.array(Image.open(ann_path))

        boxes, labels = [], []
        unique_ids = np.unique(instance_map)
        for inst_id in unique_ids:
            if inst_id < 1000:
                continue
            class_id = inst_id // 1000
            if class_id not in CITYSCAPES_LABEL_MAP:
                continue
            mask = (instance_map == inst_id)
            ys, xs = np.where(mask)
            if len(ys) < MIN_PIXELS_THRESHOLD:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(CITYSCAPES_LABEL_MAP[class_id])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        image_t = TF.to_tensor(image)
        target = {'boxes': boxes_t, 'labels': labels_t, 'image_id': torch.tensor([idx])}
        if self.transforms:
            image_t, target = self.transforms(image_t, target)
        return image_t, target

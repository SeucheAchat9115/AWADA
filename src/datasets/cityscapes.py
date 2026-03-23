import os
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Cityscapes instance ID: class_id * 1000 + instance_id
# We extract bboxes from instanceIds map: pixels where value >= 24000 (class 24+)
CITYSCAPES_LABEL_MAP = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: 5,  # train
    32: 6,  # motorcycle
    33: 7,  # bicycle
}
CLASS_NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
# Minimum number of foreground pixels for an instance to be kept as a detection box
MIN_PIXELS_THRESHOLD = 10


class CityscapesDetectionDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.samples = []

        img_base = os.path.join(root, 'leftImg8bit', split)
        ann_base = os.path.join(root, 'gtFine', split)

        for city in sorted(os.listdir(img_base)):
            city_img_dir = os.path.join(img_base, city)
            city_ann_dir = os.path.join(ann_base, city)
            if not os.path.isdir(city_img_dir):
                continue
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                stem = fname.replace('_leftImg8bit.png', '')
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
        # Extract unique instances: value = class_id * 1000 + instance_id
        unique_ids = np.unique(instance_map)
        for inst_id in unique_ids:
            if inst_id < 1000:
                continue  # not an instance (no class * 1000)
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

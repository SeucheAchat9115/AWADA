import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# Cityscapes instance ID: class_id * 1000 + instance_id
# We extract bboxes from instanceIds map: pixels where value >= 24000 (class 24+)
CITYSCAPES_LABEL_MAP = {
    24: 1,  # person
    25: 2,  # rider
    26: 3,  # car
    27: 4,  # truck
    28: 5,  # bus
    31: 6,  # train
    32: 7,  # motorcycle
    33: 8,  # bicycle
}
CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

# 7-class label map used for the Cityscapes -> BDD100k benchmark.
# The "train" class (Cityscapes ID 31) is absent from BDD100k and is excluded
# from both sides of the benchmark.  Motorcycle and bicycle are renumbered so
# that the label IDs are contiguous and match BDD100K_LABEL_MAP in bdd100k.py.
CITYSCAPES_BDD100K_LABEL_MAP = {
    24: 1,  # person
    25: 2,  # rider
    26: 3,  # car
    27: 4,  # truck
    28: 5,  # bus
    # 31 (train) excluded
    32: 6,  # motorcycle
    33: 7,  # bicycle
}
BDD100K_ALIGNED_CLASSES = ["person", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]

# Minimum number of foreground pixels for an instance to be kept as a detection box
MIN_PIXELS_THRESHOLD = 10


class CityscapesDetectionDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transforms=None,
        classes=None,
        image_root=None,
        label_map=None,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        # Allow callers to override the label map (e.g. CITYSCAPES_BDD100K_LABEL_MAP for
        # the 7-class Cityscapes → BDD100k benchmark).  Falls back to the default 8-class map.
        self._label_map = label_map if label_map is not None else CITYSCAPES_LABEL_MAP
        # Build set of allowed label indices (1-based); None means all classes
        if classes is not None:
            # Identify Cityscapes class IDs whose human-readable name is requested.
            # We always use CLASS_NAMES + CITYSCAPES_LABEL_MAP for the name lookup so that
            # the 'classes' kwarg uses stable names regardless of the label_map override.
            allowed_class_ids = {
                k
                for k in CITYSCAPES_LABEL_MAP
                if CLASS_NAMES[CITYSCAPES_LABEL_MAP[k] - 1] in classes
            }
            # Map those class IDs to labels via the (potentially overridden) label map.
            self._allowed_labels = {
                self._label_map[k]
                for k in self._label_map
                if k in allowed_class_ids
            }
        else:
            self._allowed_labels = None
        self.samples = []

        img_base = (
            image_root if image_root is not None else os.path.join(root, "leftImg8bit", split)
        )
        ann_base = os.path.join(root, "gtFine", split)

        for city in sorted(os.listdir(img_base)):
            city_img_dir = os.path.join(img_base, city)
            city_ann_dir = os.path.join(ann_base, city)
            if not os.path.isdir(city_img_dir):
                continue
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                ann_fname = stem + "_gtFine_instanceIds.png"
                ann_path = os.path.join(city_ann_dir, ann_fname)
                if os.path.exists(ann_path):
                    self.samples.append((os.path.join(city_img_dir, fname), ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        instance_map = np.array(Image.open(ann_path))

        boxes, labels = [], []
        # Extract unique instances: value = class_id * 1000 + instance_id
        unique_ids = np.unique(instance_map)
        for inst_id in unique_ids:
            if inst_id < 1000:
                continue  # not an instance (no class * 1000)
            class_id = inst_id // 1000
            if class_id not in self._label_map:
                continue
            label = self._label_map[class_id]
            if self._allowed_labels is not None and label not in self._allowed_labels:
                continue
            mask = instance_map == inst_id
            ys, xs = np.where(mask)
            if len(ys) < MIN_PIXELS_THRESHOLD:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(label)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        image_t = TF.to_tensor(image)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        if self.transforms:
            image_t, target = self.transforms(image_t, target)
        return image_t, target

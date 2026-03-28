import json
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# BDD100k detection categories aligned with the Cityscapes 7-class benchmark.
# Included classes: pedestrian, rider, car, truck, bus, motorcycle, bicycle.
# The "train" class present in Cityscapes has no reliable equivalent in BDD100k
# and is excluded from both sides of the benchmark.
BDD100K_LABEL_MAP = {
    "pedestrian": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "motorcycle": 6,
    "bicycle": 7,
}

CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]

# Minimum box dimension (width or height) below which detections are discarded
_MIN_BOX_DIM = 5


class Bdd100kDetectionDataset(Dataset):
    """BDD100k object-detection dataset aligned with the Cityscapes 7-class benchmark.

    Only the seven classes shared with Cityscapes are kept (the "train" class
    present in Cityscapes is absent from this mapping).  Labels are 1-indexed
    and match :data:`src.datasets.cityscapes.CITYSCAPES_BDD100K_LABEL_MAP`.

    Expected directory structure::

        root/
        ├── images/
        │   └── 100k/
        │       ├── train/   # JPEG images
        │       └── val/
        └── labels/
            └── det_20/
                ├── det_train.json
                └── det_val.json

    The ``image_root`` parameter lets you point to a *flat* directory of
    stylized images that use the same filenames as the originals (e.g. images
    produced by :mod:`stylize_dataset`).

    Args:
        root: Root directory of the BDD100k dataset.
        split: Dataset split, either ``"train"`` or ``"val"``.
        transforms: Optional callable ``(image_tensor, target) -> (image_tensor, target)``
            applied after loading each sample.
        image_root: Override the default ``<root>/images/100k/<split>`` image
            directory with a custom path (useful for stylized images).
    """

    def __init__(self, root, split="val", transforms=None, image_root=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.samples = []

        ann_path = os.path.join(root, "labels", "det_20", f"det_{split}.json")
        img_base = (
            image_root if image_root is not None else os.path.join(root, "images", "100k", split)
        )

        with open(ann_path) as f:
            data = json.load(f)

        for entry in data:
            fname = entry["name"]
            img_path = os.path.join(img_base, fname)
            # Pre-filter label dicts to only the categories we care about
            raw_labels = entry.get("labels") or []
            relevant = [
                lbl
                for lbl in raw_labels
                if lbl.get("category") in BDD100K_LABEL_MAP and lbl.get("box2d") is not None
            ]
            self.samples.append((img_path, relevant))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_dicts = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for lbl in label_dicts:
            box2d = lbl["box2d"]
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
            if (x2 - x1) > _MIN_BOX_DIM and (y2 - y1) > _MIN_BOX_DIM:
                boxes.append([x1, y1, x2, y2])
                labels.append(BDD100K_LABEL_MAP[lbl["category"]])

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

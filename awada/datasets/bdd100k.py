import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from awada.config import BDD100K_CLASS_NAMES as CLASS_NAMES
from awada.config import BDD100K_LABEL_MAP, MIN_BOX_DIM

__all__ = [
    "BDD100K_LABEL_MAP",
    "CLASS_NAMES",
    "MIN_BOX_DIM",
    "Bdd100kDetectionDataset",
]


class Bdd100kDetectionDataset(Dataset):
    """BDD100k object-detection dataset aligned with the Cityscapes 7-class benchmark.

    Only the seven classes shared with Cityscapes are kept (the "train" class
    present in Cityscapes is absent from this mapping).  Labels are 1-indexed
    and match :data:`awada.datasets.cityscapes.CITYSCAPES_BDD100K_LABEL_MAP`.

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

    def __init__(
        self,
        root: str,
        split: str = "val",
        transforms: Callable[..., Any] | None = None,
        image_root: str | None = None,
    ) -> None:
        """Initialise the BDD100k detection dataset.

        Args:
            root: Root directory of the BDD100k dataset.
            split: Dataset split, either ``"train"`` or ``"val"``.
            transforms: Optional callable applied after loading each sample,
                with signature ``(image_tensor, target) -> (image_tensor, target)``.
            image_root: Override the default image directory with a custom path
                (useful for stylized images that share the same filenames).
        """
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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(image_tensor, target)`` where ``image_tensor`` has
            shape ``[3, H, W]`` and ``target`` is a dict with keys
            ``"boxes"`` ``[N, 4]``, ``"labels"`` ``[N]``, and ``"image_id"`` ``[1]``.
        """
        img_path, label_dicts = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for lbl in label_dicts:
            box2d = lbl["box2d"]
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
            if (x2 - x1) > MIN_BOX_DIM and (y2 - y1) > MIN_BOX_DIM:
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

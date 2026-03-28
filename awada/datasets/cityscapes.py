import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from awada.config import (
    BDD100K_ALIGNED_CLASSES,
    CITYSCAPES_BDD100K_LABEL_MAP,
    CITYSCAPES_LABEL_MAP,
    MIN_BOX_DIM,
    MIN_PIXELS_THRESHOLD,
)
from awada.config import (
    CITYSCAPES_CLASS_NAMES as CLASS_NAMES,
)

__all__ = [
    "BDD100K_ALIGNED_CLASSES",
    "CITYSCAPES_BDD100K_LABEL_MAP",
    "CITYSCAPES_LABEL_MAP",
    "CLASS_NAMES",
    "MIN_BOX_DIM",
    "MIN_PIXELS_THRESHOLD",
    "CityscapesDetectionDataset",
]


class CityscapesDetectionDataset(Dataset):
    """Cityscapes instance-level object-detection dataset.

    Parses ``*_gtFine_instanceIds.png`` annotation files to derive bounding
    boxes from per-instance masks.  Supports an optional label-map override
    and class-name filter to target specific subsets (e.g. the 7-class
    BDD100k-aligned benchmark).

    Expected directory structure::

        root/
        ├── leftImg8bit/
        │   └── <split>/
        │       └── <city>/   # PNG images
        └── gtFine/
            └── <split>/
                └── <city>/   # *_gtFine_instanceIds.png annotation files

    Args:
        root: Root directory of the Cityscapes dataset.
        split: Dataset split (default: ``"train"``).
        transforms: Optional callable ``(image_tensor, target) -> (image_tensor, target)``
            applied after loading each sample.
        classes: Optional list of human-readable class names to keep.  When
            ``None`` (default) all classes in the label map are used.
        image_root: Override the default image directory (useful for stylized images).
        label_map: Override the default :data:`CITYSCAPES_LABEL_MAP` (e.g. use
            :data:`CITYSCAPES_BDD100K_LABEL_MAP` for the 7-class benchmark).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: object = None,
        classes: list[str] | None = None,
        image_root: str | None = None,
        label_map: dict[int, int] | None = None,
    ) -> None:
        """Initialise the Cityscapes detection dataset.

        Args:
            root: Root directory of the Cityscapes dataset.
            split: Dataset split (e.g. ``"train"`` or ``"val"``).
            transforms: Optional callable applied after loading each sample.
            classes: Optional list of class names to retain.
            image_root: Override the default image directory.
            label_map: Override the default instance-ID-to-label map.
        """
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
                self._label_map[k] for k in self._label_map if k in allowed_class_ids
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
            if (x2 - x1) > MIN_BOX_DIM and (y2 - y1) > MIN_BOX_DIM:
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

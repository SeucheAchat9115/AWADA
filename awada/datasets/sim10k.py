import os
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from awada.config import MIN_BOX_DIM
from awada.config import SIM10K_CLASS_NAMES as CLASS_NAMES

__all__ = [
    "CLASS_NAMES",
    "Sim10kDetectionDataset",
]


class Sim10kDetectionDataset(Dataset):
    """Driving in the Matrix (sim10k) synthetic driving dataset for object detection.

    Annotations are in PASCAL VOC XML format and contain only the 'car' class.
    All images are used without any train/val split.

    Expected directory structure::

        root/
        ├── images/       # JPEG images (e.g. 00001.jpg)
        └── Annotations/  # PASCAL VOC XML files (e.g. 00001.xml)
    """

    def __init__(self, root: str, transforms: object = None, image_dir: str | None = None) -> None:
        """Initialise the Sim10k detection dataset.

        Args:
            root: Root directory containing ``images/`` and ``Annotations/``.
            transforms: Optional callable ``(image_tensor, target) -> (image_tensor, target)``
                applied after loading each sample.
            image_dir: Override the default ``<root>/images`` directory.
        """
        self.root = root
        self.transforms = transforms
        self.image_dir = image_dir if image_dir is not None else os.path.join(root, "images")
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(image_tensor, target)`` where ``image_tensor`` has
            shape ``[3, H, W]`` and ``target`` is a dict with keys
            ``"boxes"`` ``[N, 4]``, ``"labels"`` ``[N]``, and ``"image_id"`` ``[1]``.
        """
        fname = self.image_files[idx]
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(self.image_dir, fname)
        ann_path = os.path.join(self.root, "Annotations", stem + ".xml")

        image = Image.open(img_path).convert("RGB")

        boxes, labels = self._parse_annotation(ann_path)

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

    @staticmethod
    def _parse_annotation(ann_path: str) -> tuple[list[list[float]], list[int]]:
        """Parse a PASCAL VOC XML annotation file and return boxes and labels."""
        boxes, labels = [], []
        if not os.path.exists(ann_path):
            return boxes, labels
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name != "car":
                continue
            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
            if (x2 - x1) > MIN_BOX_DIM and (y2 - y1) > MIN_BOX_DIM:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_NAMES.index("car") + 1)
        return boxes, labels

import csv
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
    "generate_det_json",
]


def _load_scalabel_annotations(ann_path: str) -> list[dict[str, Any]]:
    """Load BDD100K annotations from the full scalabel labels file.

    The full scalabel labels file (e.g. ``bdd100k_labels_images_train.json``)
    contains all annotation types including detection, segmentation and
    tracking.  Each entry has the same ``name`` / ``labels`` structure as the
    ``det_20`` JSON, so extra fields are simply ignored by the caller.

    Args:
        ann_path: Path to the scalabel JSON file.

    Returns:
        List of annotation entries compatible with the det_20 format.
    """
    with open(ann_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _load_csv_annotations(ann_path: str) -> list[dict[str, Any]]:
    """Load BDD100K annotations from a CSV file (non-JSON raw format).

    The CSV file must contain a header row with the columns
    ``name``, ``category``, ``x1``, ``y1``, ``x2``, ``y2``.
    Each subsequent row represents one bounding-box annotation.  Multiple
    rows may share the same ``name`` (one row per box).

    Example CSV content::

        name,category,x1,y1,x2,y2
        image001.jpg,car,5.0,5.0,40.0,40.0
        image001.jpg,pedestrian,100.0,50.0,130.0,150.0
        image002.jpg,truck,20.0,30.0,100.0,80.0

    Args:
        ann_path: Path to the CSV annotation file.

    Returns:
        List of annotation entries compatible with the det_20 format.
    """
    entries: dict[str, dict[str, Any]] = {}
    with open(ann_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if name not in entries:
                entries[name] = {"name": name, "labels": []}
            entries[name]["labels"].append(
                {
                    "category": row["category"],
                    "box2d": {
                        "x1": float(row["x1"]),
                        "y1": float(row["y1"]),
                        "x2": float(row["x2"]),
                        "y2": float(row["y2"]),
                    },
                }
            )
    return list(entries.values())


def _resolve_annotations(root: str, split: str) -> list[dict[str, Any]]:
    """Locate and load annotations for *split*, trying multiple formats.

    The search order is:

    1. ``labels/det_20/det_{split}.json`` — pre-generated det_20 JSON (fastest).
    2. ``labels/bdd100k_labels_images_{split}.json`` — full scalabel labels
       downloaded directly from the BDD100K website.
    3. ``labels/det_20/det_{split}.csv`` — flat CSV file, one bounding-box
       per row (non-JSON raw format).

    Args:
        root: Root directory of the BDD100K dataset.
        split: Dataset split, ``"train"`` or ``"val"``.

    Returns:
        List of annotation entries (det_20-compatible format).

    Raises:
        FileNotFoundError: When no annotation file is found in any of the
            expected locations.
    """
    det_json = os.path.join(root, "labels", "det_20", f"det_{split}.json")
    scalabel_json = os.path.join(root, "labels", f"bdd100k_labels_images_{split}.json")
    csv_path = os.path.join(root, "labels", "det_20", f"det_{split}.csv")

    if os.path.exists(det_json):
        with open(det_json) as f:
            return json.load(f)  # type: ignore[no-any-return]
    if os.path.exists(scalabel_json):
        return _load_scalabel_annotations(scalabel_json)
    if os.path.exists(csv_path):
        return _load_csv_annotations(csv_path)

    raise FileNotFoundError(
        f"No annotation file found for split '{split}'. "
        f"Expected one of:\n"
        f"  {det_json}  (det_20 JSON — pre-generated)\n"
        f"  {scalabel_json}  (full scalabel labels from BDD100K website)\n"
        f"  {csv_path}  (CSV raw format: name,category,x1,y1,x2,y2)"
    )


def generate_det_json(raw_ann_path: str, output_path: str) -> None:
    """Generate a ``det_20``-compatible JSON file from a raw annotation file.

    This is a convenience utility for users who want to pre-generate the
    filtered det_20 JSON (e.g. for faster subsequent loads) from either the
    full scalabel labels file or a CSV annotation file.  The output JSON
    contains only entries and labels that are compatible with the det_20
    format (i.e. have a ``name``, ``category`` in
    :data:`~awada.config.BDD100K_LABEL_MAP`, and a valid ``box2d``).

    Args:
        raw_ann_path: Path to the raw annotation file.  Supported formats are
            the full BDD100K scalabel JSON (``bdd100k_labels_images_*.json``)
            and the CSV format (``*.csv``).
        output_path: Destination path for the generated det_20 JSON file.
            Parent directories are created automatically.

    Example::

        from awada.datasets.bdd100k import generate_det_json

        generate_det_json(
            "/data/bdd100k/labels/bdd100k_labels_images_train.json",
            "/data/bdd100k/labels/det_20/det_train.json",
        )
    """
    if raw_ann_path.endswith(".csv"):
        data = _load_csv_annotations(raw_ann_path)
    else:
        data = _load_scalabel_annotations(raw_ann_path)

    # Filter to only entries/labels relevant for the 7-class detection task.
    filtered: list[dict[str, Any]] = []
    for entry in data:
        raw_labels = entry.get("labels") or []
        relevant = [
            {"category": lbl["category"], "box2d": lbl["box2d"]}
            for lbl in raw_labels
            if lbl.get("category") in BDD100K_LABEL_MAP and lbl.get("box2d") is not None
        ]
        filtered.append({"name": entry["name"], "labels": relevant})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f)


class Bdd100kDetectionDataset(Dataset):
    """BDD100k object-detection dataset aligned with the Cityscapes 7-class benchmark.

    Only the seven classes shared with Cityscapes are kept (the "train" class
    present in Cityscapes is absent from this mapping).  Labels are 1-indexed
    and match :data:`src.datasets.cityscapes.CITYSCAPES_BDD100K_LABEL_MAP`.

    The dataset automatically detects the annotation format and loads it
    without any manual pre-processing.  The supported formats, in lookup
    order, are:

    1. **det_20 JSON** — ``labels/det_20/det_{split}.json`` (fastest).
    2. **Scalabel full labels** — ``labels/bdd100k_labels_images_{split}.json``
       (the raw labels downloaded directly from the BDD100K website).
    3. **CSV raw format** — ``labels/det_20/det_{split}.csv``
       (non-JSON flat file: ``name,category,x1,y1,x2,y2``).

    Minimal directory layout (det_20 JSON format)::

        root/
        ├── images/
        │   └── 100k/
        │       ├── train/   # JPEG images
        │       └── val/
        └── labels/
            └── det_20/
                ├── det_train.json
                └── det_val.json

    Alternative layout using the full scalabel labels (raw download)::

        root/
        ├── images/
        │   └── 100k/
        │       ├── train/
        │       └── val/
        └── labels/
            ├── bdd100k_labels_images_train.json
            └── bdd100k_labels_images_val.json

    Alternative layout using the CSV raw format::

        root/
        ├── images/
        │   └── 100k/
        │       ├── train/
        │       └── val/
        └── labels/
            └── det_20/
                ├── det_train.csv
                └── det_val.csv

    The ``image_root`` parameter lets you point to a *flat* directory of
    stylized images that use the same filenames as the originals (e.g. images
    produced by :mod:`stylize_dataset`).

    Use :func:`generate_det_json` to pre-generate a det_20 JSON from any of
    the raw formats for faster subsequent loading.

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

        img_base = (
            image_root if image_root is not None else os.path.join(root, "images", "100k", split)
        )

        data = _resolve_annotations(root, split)

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

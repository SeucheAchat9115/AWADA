"""Tests for the Bdd100kDetectionDataset class and related constants."""

import json

import numpy as np
from PIL import Image

from src.datasets.bdd100k import BDD100K_LABEL_MAP, CLASS_NAMES, Bdd100kDetectionDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestBdd100kConstants:
    def test_label_map_keys_are_strings(self):
        for k in BDD100K_LABEL_MAP:
            assert isinstance(k, str), f"Expected str key, got {type(k)}"

    def test_label_map_values_are_positive(self):
        for v in BDD100K_LABEL_MAP.values():
            assert v >= 1, f"Labels must be 1-indexed, got {v}"

    def test_label_map_values_unique(self):
        values = list(BDD100K_LABEL_MAP.values())
        assert len(values) == len(set(values)), "Label IDs must be unique"

    def test_class_names_length_matches_label_map(self):
        assert len(CLASS_NAMES) == len(BDD100K_LABEL_MAP)

    def test_seven_aligned_classes(self):
        """Benchmark uses exactly 7 classes (Cityscapes 8 minus train)."""
        assert len(CLASS_NAMES) == 7

    def test_expected_classes_present(self):
        for cls in ("pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"):
            assert cls in BDD100K_LABEL_MAP, f"Missing expected class: {cls}"

    def test_train_class_absent(self):
        """The 'train' class has no BDD100k equivalent and must not appear."""
        assert "train" not in BDD100K_LABEL_MAP

    def test_pedestrian_label_is_one(self):
        assert BDD100K_LABEL_MAP["pedestrian"] == 1

    def test_label_ids_contiguous_starting_at_one(self):
        values = sorted(BDD100K_LABEL_MAP.values())
        assert values == list(range(1, len(values) + 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bdd100k_root(tmp_path, split="val", entries=None):
    """Create a minimal BDD100k directory structure with synthetic data.

    ``entries`` is a list of dicts with keys ``name`` and ``labels``
    (BDD100k annotation format).  If *None*, a single image with one car
    annotation is created.
    """
    img_dir = tmp_path / "images" / "100k" / split
    label_dir = tmp_path / "labels" / "det_20"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    if entries is None:
        fname = "sample_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))
        entries = [
            {
                "name": fname,
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0},
                    }
                ],
            }
        ]
    else:
        for entry in entries:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(str(img_dir / entry["name"]))

    ann_path = label_dir / f"det_{split}.json"
    with open(str(ann_path), "w") as f:
        json.dump(entries, f)

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestBdd100kDetectionDataset:
    def test_len(self, tmp_path):
        root = _make_bdd100k_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 1

    def test_getitem_returns_image_and_target(self, tmp_path):
        root = _make_bdd100k_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        image, target = ds[0]
        assert image.shape[0] == 3  # RGB channels

    def test_getitem_target_keys(self, tmp_path):
        root = _make_bdd100k_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target

    def test_car_box_detected(self, tmp_path):
        root = _make_bdd100k_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["car"]

    def test_label_values_correct(self, tmp_path):
        """Each class in the label map should receive the correct 1-based label."""
        entries = []
        for cat in BDD100K_LABEL_MAP:
            fname = f"{cat}.jpg"
            entries.append(
                {
                    "name": fname,
                    "labels": [
                        {"category": cat, "box2d": {"x1": 1.0, "y1": 1.0, "x2": 30.0, "y2": 30.0}}
                    ],
                }
            )
        root = _make_bdd100k_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        for idx, cat in enumerate(BDD100K_LABEL_MAP):
            _, target = ds[idx]
            assert target["labels"][0].item() == BDD100K_LABEL_MAP[cat], f"Wrong label for {cat}"

    def test_unknown_category_excluded(self, tmp_path):
        """Annotations whose category is not in BDD100K_LABEL_MAP should be ignored."""
        entries = [
            {
                "name": "img.jpg",
                "labels": [
                    {
                        "category": "traffic light",
                        "box2d": {"x1": 1.0, "y1": 1.0, "x2": 30.0, "y2": 30.0},
                    },
                    {
                        "category": "car",
                        "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0},
                    },
                ],
            }
        ]
        root = _make_bdd100k_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        # Only the car should be kept
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["car"]

    def test_tiny_box_excluded(self, tmp_path):
        """Boxes with width or height <= 5 should be discarded."""
        entries = [
            {
                "name": "img.jpg",
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 10.0, "y1": 10.0, "x2": 13.0, "y2": 13.0},
                    }
                ],
            }
        ]
        root = _make_bdd100k_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 0

    def test_null_labels_gives_empty_boxes(self, tmp_path):
        """Images with null labels should yield zero annotations."""
        entries = [{"name": "img.jpg", "labels": None}]
        root = _make_bdd100k_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)

    def test_image_root_override(self, tmp_path):
        """image_root should allow loading images from a custom directory."""
        # Create the standard annotation structure
        label_dir = tmp_path / "labels" / "det_20"
        label_dir.mkdir(parents=True)
        fname = "img.jpg"
        entries = [
            {
                "name": fname,
                "labels": [
                    {"category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}}
                ],
            }
        ]
        with open(str(label_dir / "det_val.json"), "w") as f:
            json.dump(entries, f)

        # Place the image in a custom directory (not the default one)
        custom_img_dir = tmp_path / "stylized_images"
        custom_img_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(custom_img_dir / fname))

        ds = Bdd100kDetectionDataset(str(tmp_path), split="val", image_root=str(custom_img_dir))
        assert len(ds) == 1
        image, target = ds[0]
        assert image.shape[0] == 3

    def test_transforms_applied(self, tmp_path):
        """The transforms callable should be invoked on each sample."""
        root = _make_bdd100k_root(tmp_path)
        call_log = []

        def record_transform(img, tgt):
            call_log.append(True)
            return img, tgt

        ds = Bdd100kDetectionDataset(root, split="val", transforms=record_transform)
        ds[0]
        assert len(call_log) == 1

    def test_multiple_entries(self, tmp_path):
        """Dataset length should match the number of entries in the JSON."""
        entries = [
            {
                "name": f"img{i}.jpg",
                "labels": [
                    {
                        "category": "pedestrian",
                        "box2d": {"x1": 1.0, "y1": 1.0, "x2": 20.0, "y2": 20.0},
                    }
                ],
            }
            for i in range(5)
        ]
        root = _make_bdd100k_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# Class alignment with Cityscapes
# ---------------------------------------------------------------------------


class TestCityscapesBdd100kAlignment:
    def test_aligned_label_ids_match(self):
        """Cityscapes BDD100k label map and BDD100k label map must use the same IDs."""
        from src.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        cs_ids = set(CITYSCAPES_BDD100K_LABEL_MAP.values())
        bdd_ids = set(BDD100K_LABEL_MAP.values())
        assert cs_ids == bdd_ids, f"Label ID sets differ: Cityscapes={cs_ids}, BDD100k={bdd_ids}"

    def test_same_number_of_classes(self):
        from src.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        assert len(CITYSCAPES_BDD100K_LABEL_MAP) == len(BDD100K_LABEL_MAP) == 7

    def test_cityscapes_bdd100k_map_excludes_train(self):
        from src.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        # Cityscapes class ID 31 is "train"; it must not appear in the aligned map
        assert 31 not in CITYSCAPES_BDD100K_LABEL_MAP

"""Tests for the Bdd100kDetectionDataset class and related constants."""

import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from awada.datasets.bdd100k import (
    BDD100K_LABEL_MAP,
    CLASS_NAMES,
    Bdd100kDetectionDataset,
    generate_det_json,
)

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
        from awada.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        cs_ids = set(CITYSCAPES_BDD100K_LABEL_MAP.values())
        bdd_ids = set(BDD100K_LABEL_MAP.values())
        assert cs_ids == bdd_ids, f"Label ID sets differ: Cityscapes={cs_ids}, BDD100k={bdd_ids}"

    def test_same_number_of_classes(self):
        from awada.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        assert len(CITYSCAPES_BDD100K_LABEL_MAP) == len(BDD100K_LABEL_MAP) == 7

    def test_cityscapes_bdd100k_map_excludes_train(self):
        from awada.datasets.cityscapes import CITYSCAPES_BDD100K_LABEL_MAP

        # Cityscapes class ID 31 is "train"; it must not appear in the aligned map
        assert 31 not in CITYSCAPES_BDD100K_LABEL_MAP


# ---------------------------------------------------------------------------
# Helpers for raw-format tests
# ---------------------------------------------------------------------------


def _make_scalabel_root(tmp_path, split="val", entries=None):
    """Create a BDD100k root with full scalabel labels (bdd100k_labels_images_{split}.json).

    The scalabel format mirrors the det_20 structure but allows extra fields
    per label (attributes, manualShape, etc.).  If *entries* is None, a
    single image with one car annotation (with extra scalabel fields) is
    created.
    """
    img_dir = tmp_path / "images" / "100k" / split
    label_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    if entries is None:
        fname = "sample_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))
        entries = [
            {
                "name": fname,
                "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
                "labels": [
                    {
                        "id": 1,
                        "category": "car",
                        "attributes": {"occluded": False, "truncated": False},
                        "manualShape": True,
                        "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0},
                        "poly2d": None,
                        "box3d": None,
                    }
                ],
            }
        ]
    else:
        for entry in entries:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(str(img_dir / entry["name"]))

    ann_path = label_dir / f"bdd100k_labels_images_{split}.json"
    with open(str(ann_path), "w") as f:
        json.dump(entries, f)

    return str(tmp_path)


def _make_csv_root(tmp_path, split="val", rows=None):
    """Create a BDD100k root with CSV raw annotations.

    *rows* is a list of dicts with keys ``name``, ``category``,
    ``x1``, ``y1``, ``x2``, ``y2``.  If *None*, a single image with one car
    annotation is used.
    """
    img_dir = tmp_path / "images" / "100k" / split
    label_dir = tmp_path / "labels" / "det_20"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    if rows is None:
        fname = "sample_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))
        rows = [{"name": fname, "category": "car", "x1": "5.0", "y1": "5.0", "x2": "40.0", "y2": "40.0"}]
    else:
        seen = set()
        for row in rows:
            if row["name"] not in seen:
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(str(img_dir / row["name"]))
                seen.add(row["name"])

    csv_path = label_dir / f"det_{split}.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Scalabel raw format loading
# ---------------------------------------------------------------------------


class TestBdd100kScalabelFormat:
    """Tests for loading BDD100k from the full scalabel labels format."""

    def test_loads_from_scalabel_json(self, tmp_path):
        """Dataset should load successfully from bdd100k_labels_images_{split}.json."""
        root = _make_scalabel_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 1

    def test_scalabel_car_detected(self, tmp_path):
        """Car annotation from scalabel format should be correctly loaded."""
        root = _make_scalabel_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["car"]

    def test_scalabel_extra_fields_ignored(self, tmp_path):
        """Extra scalabel fields (id, attributes, poly2d, etc.) should not cause errors."""
        root = _make_scalabel_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        image, target = ds[0]
        assert image.shape[0] == 3

    def test_scalabel_non_detection_labels_skipped(self, tmp_path):
        """Labels without box2d (segmentation-only labels) should be ignored."""
        fname = "img.jpg"
        entries = [
            {
                "name": fname,
                "attributes": {},
                "labels": [
                    # segmentation-only label: no box2d
                    {"id": 1, "category": "car", "box2d": None, "poly2d": [{"vertices": []}]},
                    # valid detection label
                    {"id": 2, "category": "pedestrian", "box2d": {"x1": 1.0, "y1": 1.0, "x2": 30.0, "y2": 30.0}},
                ],
            }
        ]
        root = _make_scalabel_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["pedestrian"]

    def test_scalabel_unknown_category_excluded(self, tmp_path):
        """Non-benchmark categories (e.g. traffic light) in scalabel format must be ignored."""
        fname = "img.jpg"
        entries = [
            {
                "name": fname,
                "attributes": {},
                "labels": [
                    {"id": 1, "category": "traffic light", "box2d": {"x1": 1.0, "y1": 1.0, "x2": 20.0, "y2": 20.0}},
                    {"id": 2, "category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}},
                ],
            }
        ]
        root = _make_scalabel_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["car"]

    def test_scalabel_multiple_images(self, tmp_path):
        """Dataset length must match the number of images in the scalabel file."""
        entries = [
            {
                "name": f"img{i}.jpg",
                "attributes": {},
                "labels": [
                    {"id": 1, "category": "car", "box2d": {"x1": 1.0, "y1": 1.0, "x2": 30.0, "y2": 30.0}}
                ],
            }
            for i in range(4)
        ]
        root = _make_scalabel_root(tmp_path, entries=entries)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 4


# ---------------------------------------------------------------------------
# CSV raw format loading
# ---------------------------------------------------------------------------


class TestBdd100kCsvFormat:
    """Tests for loading BDD100k from the non-JSON CSV raw annotation format."""

    def test_loads_from_csv(self, tmp_path):
        """Dataset should load successfully from det_{split}.csv."""
        root = _make_csv_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 1

    def test_csv_car_detected(self, tmp_path):
        """Car annotation from CSV format should be correctly loaded."""
        root = _make_csv_root(tmp_path)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["car"]

    def test_csv_multiple_boxes_same_image(self, tmp_path):
        """Multiple CSV rows for the same image should be combined into one dataset entry."""
        rows = [
            {"name": "img.jpg", "category": "car", "x1": "5.0", "y1": "5.0", "x2": "40.0", "y2": "40.0"},
            {"name": "img.jpg", "category": "pedestrian", "x1": "50.0", "y1": "50.0", "x2": "80.0", "y2": "100.0"},
        ]
        root = _make_csv_root(tmp_path, rows=rows)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 1
        _, target = ds[0]
        assert target["boxes"].shape[0] == 2

    def test_csv_multiple_images(self, tmp_path):
        """Each unique image name in the CSV must become one dataset entry."""
        rows = [
            {"name": f"img{i}.jpg", "category": "car", "x1": "5.0", "y1": "5.0", "x2": "40.0", "y2": "40.0"}
            for i in range(3)
        ]
        root = _make_csv_root(tmp_path, rows=rows)
        ds = Bdd100kDetectionDataset(root, split="val")
        assert len(ds) == 3

    def test_csv_box_coordinates_parsed_correctly(self, tmp_path):
        """Box coordinates in the CSV should be accurately read as floats."""
        rows = [
            {"name": "img.jpg", "category": "truck", "x1": "10.5", "y1": "20.5", "x2": "60.5", "y2": "80.5"}
        ]
        root = _make_csv_root(tmp_path, rows=rows)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        box = target["boxes"][0].tolist()
        assert box == pytest.approx([10.5, 20.5, 60.5, 80.5])

    def test_csv_unknown_category_excluded(self, tmp_path):
        """Non-benchmark categories in the CSV must be filtered out."""
        rows = [
            {"name": "img.jpg", "category": "traffic sign", "x1": "1.0", "y1": "1.0", "x2": "30.0", "y2": "30.0"},
            {"name": "img.jpg", "category": "bus", "x1": "5.0", "y1": "5.0", "x2": "40.0", "y2": "40.0"},
        ]
        root = _make_csv_root(tmp_path, rows=rows)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == BDD100K_LABEL_MAP["bus"]

    def test_csv_tiny_box_excluded(self, tmp_path):
        """CSV boxes with width or height <= MIN_BOX_DIM should be discarded."""
        rows = [
            {"name": "img.jpg", "category": "car", "x1": "10.0", "y1": "10.0", "x2": "13.0", "y2": "13.0"}
        ]
        root = _make_csv_root(tmp_path, rows=rows)
        ds = Bdd100kDetectionDataset(root, split="val")
        _, target = ds[0]
        assert target["boxes"].shape[0] == 0


# ---------------------------------------------------------------------------
# Format fallback order and missing-file errors
# ---------------------------------------------------------------------------


class TestBdd100kFormatFallback:
    """Tests for format detection priority and error handling."""

    def test_det20_json_takes_priority_over_scalabel(self, tmp_path):
        """det_20 JSON must be used when present even if scalabel file also exists."""
        # Create det_20 JSON with 1 image
        img_dir = tmp_path / "images" / "100k" / "val"
        img_dir.mkdir(parents=True)
        fname = "det20.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))

        det20_dir = tmp_path / "labels" / "det_20"
        det20_dir.mkdir(parents=True)
        det20_entries = [{"name": fname, "labels": [{"category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}}]}]
        with open(str(det20_dir / "det_val.json"), "w") as f:
            json.dump(det20_entries, f)

        # Also create a scalabel file with 3 images (should be ignored)
        scalabel_entries = [
            {"name": f"scalabel{i}.jpg", "attributes": {}, "labels": []} for i in range(3)
        ]
        with open(str(tmp_path / "labels" / "bdd100k_labels_images_val.json"), "w") as f:
            json.dump(scalabel_entries, f)

        ds = Bdd100kDetectionDataset(str(tmp_path), split="val")
        # Should use det_20 JSON (1 image), not scalabel file (3 images)
        assert len(ds) == 1

    def test_scalabel_takes_priority_over_csv(self, tmp_path):
        """Scalabel JSON must be used when present even if CSV also exists."""
        img_dir = tmp_path / "images" / "100k" / "val"
        img_dir.mkdir(parents=True)
        fname = "scal.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))

        # Scalabel file with 1 image
        label_dir = tmp_path / "labels"
        label_dir.mkdir(parents=True)
        scalabel_entries = [{"name": fname, "attributes": {}, "labels": [{"id": 1, "category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}}]}]
        with open(str(label_dir / "bdd100k_labels_images_val.json"), "w") as f:
            json.dump(scalabel_entries, f)

        # CSV file with 2 distinct images (should be ignored)
        det20_dir = tmp_path / "labels" / "det_20"
        det20_dir.mkdir(parents=True)
        for extra_fname in ["csv1.jpg", "csv2.jpg"]:
            img2 = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img2.save(str(img_dir / extra_fname))
        with open(str(det20_dir / "det_val.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "category", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for n in ["csv1.jpg", "csv2.jpg"]:
                writer.writerow({"name": n, "category": "car", "x1": 5, "y1": 5, "x2": 40, "y2": 40})

        ds = Bdd100kDetectionDataset(str(tmp_path), split="val")
        # Should use scalabel file (1 image), not CSV (2 images)
        assert len(ds) == 1

    def test_missing_annotations_raises_file_not_found(self, tmp_path):
        """FileNotFoundError must be raised when no annotation file exists."""
        img_dir = tmp_path / "images" / "100k" / "val"
        img_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            Bdd100kDetectionDataset(str(tmp_path), split="val")


# ---------------------------------------------------------------------------
# generate_det_json utility
# ---------------------------------------------------------------------------


class TestGenerateDetJson:
    """Tests for the generate_det_json() public utility."""

    def test_generate_from_scalabel(self, tmp_path):
        """generate_det_json should produce a valid det_20 JSON from scalabel input."""
        scalabel_entries = [
            {
                "name": "a.jpg",
                "attributes": {},
                "labels": [
                    {"id": 1, "category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}, "poly2d": None},
                    {"id": 2, "category": "traffic light", "box2d": {"x1": 1.0, "y1": 1.0, "x2": 20.0, "y2": 20.0}},
                ],
            }
        ]
        raw_path = str(tmp_path / "bdd100k_labels_images_val.json")
        with open(raw_path, "w") as f:
            json.dump(scalabel_entries, f)

        output_path = str(tmp_path / "det_20" / "det_val.json")
        generate_det_json(raw_path, output_path)

        with open(output_path) as f:
            result = json.load(f)

        assert len(result) == 1
        assert result[0]["name"] == "a.jpg"
        # Only the car should be kept (traffic light is not in BDD100K_LABEL_MAP)
        assert len(result[0]["labels"]) == 1
        assert result[0]["labels"][0]["category"] == "car"

    def test_generate_from_csv(self, tmp_path):
        """generate_det_json should produce a valid det_20 JSON from a CSV file."""
        csv_path = str(tmp_path / "det_val.csv")
        rows = [
            {"name": "b.jpg", "category": "pedestrian", "x1": "1.0", "y1": "1.0", "x2": "30.0", "y2": "30.0"},
            {"name": "b.jpg", "category": "traffic sign", "x1": "5.0", "y1": "5.0", "x2": "20.0", "y2": "20.0"},
            {"name": "c.jpg", "category": "truck", "x1": "10.0", "y1": "10.0", "x2": "50.0", "y2": "50.0"},
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "category", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(rows)

        output_path = str(tmp_path / "out" / "det_val.json")
        generate_det_json(csv_path, output_path)

        with open(output_path) as f:
            result = json.load(f)

        names = {e["name"] for e in result}
        assert names == {"b.jpg", "c.jpg"}
        b_entry = next(e for e in result if e["name"] == "b.jpg")
        # Only pedestrian survives (traffic sign is not in BDD100K_LABEL_MAP)
        assert len(b_entry["labels"]) == 1
        assert b_entry["labels"][0]["category"] == "pedestrian"

    def test_generate_creates_parent_dirs(self, tmp_path):
        """generate_det_json must create missing parent directories for output_path."""
        scalabel_entries = [{"name": "x.jpg", "attributes": {}, "labels": []}]
        raw_path = str(tmp_path / "raw.json")
        with open(raw_path, "w") as f:
            json.dump(scalabel_entries, f)

        deep_output = str(tmp_path / "a" / "b" / "c" / "det_val.json")
        generate_det_json(raw_path, deep_output)
        assert os.path.exists(deep_output)

    def test_generate_output_loadable_by_dataset(self, tmp_path):
        """JSON generated by generate_det_json must be loadable by Bdd100kDetectionDataset."""
        img_dir = tmp_path / "images" / "100k" / "val"
        img_dir.mkdir(parents=True)
        fname = "gen.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))

        scalabel_entries = [
            {
                "name": fname,
                "attributes": {},
                "labels": [
                    {"id": 1, "category": "car", "box2d": {"x1": 5.0, "y1": 5.0, "x2": 40.0, "y2": 40.0}}
                ],
            }
        ]
        raw_path = str(tmp_path / "labels" / "bdd100k_labels_images_val.json")
        (tmp_path / "labels").mkdir(parents=True)
        with open(raw_path, "w") as f:
            json.dump(scalabel_entries, f)

        det20_dir = tmp_path / "labels" / "det_20"
        det20_dir.mkdir(parents=True)
        output_path = str(det20_dir / "det_val.json")
        generate_det_json(raw_path, output_path)

        ds = Bdd100kDetectionDataset(str(tmp_path), split="val")
        assert len(ds) == 1
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1

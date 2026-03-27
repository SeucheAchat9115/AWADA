"""Tests for Cityscapes and FoggyCityscapes dataset constants, label maps, and dataset loading."""

import numpy as np
from PIL import Image

from src.datasets.cityscapes import (
    BDD100K_ALIGNED_CLASSES,
    CITYSCAPES_BDD100K_LABEL_MAP,
    CITYSCAPES_LABEL_MAP,
    CLASS_NAMES,
    MIN_PIXELS_THRESHOLD,
)
from src.datasets.foggy_cityscapes import FoggyCityscapesDataset


class TestCityscapesConstants:
    def test_label_map_keys_are_ints(self):
        for k in CITYSCAPES_LABEL_MAP:
            assert isinstance(k, int), f"Expected int key, got {type(k)}"

    def test_label_map_values_are_positive(self):
        for v in CITYSCAPES_LABEL_MAP.values():
            assert v >= 1, f"Labels must be 1-indexed, got {v}"

    def test_class_names_not_empty(self):
        assert len(CLASS_NAMES) > 0

    def test_label_map_values_within_class_names_range(self):
        for v in CITYSCAPES_LABEL_MAP.values():
            assert 1 <= v <= len(CLASS_NAMES)

    def test_min_pixels_threshold_positive(self):
        assert MIN_PIXELS_THRESHOLD > 0

    def test_expected_classes_present(self):
        """Known Cityscapes traffic classes should appear in CLASS_NAMES."""
        for cls in ("car", "person", "bus", "truck"):
            assert cls in CLASS_NAMES, f"Missing expected class: {cls}"

    def test_person_label(self):
        assert CITYSCAPES_LABEL_MAP[24] == 1  # person

    def test_car_label(self):
        assert CITYSCAPES_LABEL_MAP[26] == 3  # car

    def test_label_map_all_unique_values(self):
        values = list(CITYSCAPES_LABEL_MAP.values())
        assert len(values) == len(set(values)), "Label map values should be unique"


class TestCityscapesBdd100kLabelMap:
    def test_seven_classes(self):
        assert len(CITYSCAPES_BDD100K_LABEL_MAP) == 7

    def test_train_class_excluded(self):
        """Cityscapes class ID 31 (train) must not appear in the BDD100k-aligned map."""
        assert 31 not in CITYSCAPES_BDD100K_LABEL_MAP

    def test_all_other_cityscapes_ids_present(self):
        expected_ids = {24, 25, 26, 27, 28, 32, 33}
        assert set(CITYSCAPES_BDD100K_LABEL_MAP.keys()) == expected_ids

    def test_label_ids_contiguous_starting_at_one(self):
        values = sorted(CITYSCAPES_BDD100K_LABEL_MAP.values())
        assert values == list(range(1, len(values) + 1))

    def test_motorcycle_remapped_to_six(self):
        assert CITYSCAPES_BDD100K_LABEL_MAP[32] == 6  # motorcycle

    def test_bicycle_remapped_to_seven(self):
        assert CITYSCAPES_BDD100K_LABEL_MAP[33] == 7  # bicycle

    def test_bdd100k_aligned_classes_length(self):
        assert len(BDD100K_ALIGNED_CLASSES) == 7

    def test_bdd100k_aligned_classes_no_train(self):
        assert "train" not in BDD100K_ALIGNED_CLASSES


class TestCityscapesDetectionDataset:
    def _make_cityscapes_root(self, tmp_path, split="train"):
        """Create a minimal Cityscapes directory structure with synthetic data."""
        img_dir = tmp_path / "leftImg8bit" / split / "aachen"
        ann_dir = tmp_path / "gtFine" / split / "aachen"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        # Image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / "aachen_000000_000019_leftImg8bit.png"))

        # Instance map: create a single car instance (class 26, instance 26001)
        instance_map = np.zeros((64, 64), dtype=np.uint16)
        # Fill a 30x30 region with instance id 26001 (class 26 = car)
        instance_map[5:35, 5:35] = 26001
        inst_img = Image.fromarray(instance_map)
        inst_img.save(str(ann_dir / "aachen_000000_000019_gtFine_instanceIds.png"))

        return str(tmp_path)

    def test_len(self, tmp_path):
        from src.datasets.cityscapes import CityscapesDetectionDataset

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(root, split="train")
        assert len(ds) == 1

    def test_getitem_returns_image_and_target(self, tmp_path):
        from src.datasets.cityscapes import CityscapesDetectionDataset

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(root, split="train")
        image, target = ds[0]
        assert image.shape[0] == 3

    def test_getitem_target_keys(self, tmp_path):
        from src.datasets.cityscapes import CityscapesDetectionDataset

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(root, split="train")
        _, target = ds[0]
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target

    def test_car_box_detected(self, tmp_path):
        from src.datasets.cityscapes import CityscapesDetectionDataset

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(root, split="train")
        _, target = ds[0]
        # There should be exactly one car box
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == CITYSCAPES_LABEL_MAP[26]  # car

    def test_class_filter(self, tmp_path):
        """When classes=['person'] is specified, car annotations should be excluded."""
        from src.datasets.cityscapes import CityscapesDetectionDataset

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(root, split="train", classes=["person"])
        _, target = ds[0]
        assert target["boxes"].shape[0] == 0  # no person instances in our synthetic data

    def test_label_map_override_remaps_car_label(self, tmp_path):
        """When label_map=CITYSCAPES_BDD100K_LABEL_MAP, car (class 26) should get label 3."""
        from src.datasets.cityscapes import (
            CITYSCAPES_BDD100K_LABEL_MAP,
            CityscapesDetectionDataset,
        )

        root = self._make_cityscapes_root(tmp_path)
        ds = CityscapesDetectionDataset(
            root, split="train", label_map=CITYSCAPES_BDD100K_LABEL_MAP
        )
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        # car (class 26) → label 3 in both maps
        assert target["labels"][0].item() == CITYSCAPES_BDD100K_LABEL_MAP[26]

    def _make_cityscapes_root_with_train_instance(self, tmp_path, split="train"):
        """Create a Cityscapes root that has both a car and a train instance."""
        img_dir = tmp_path / "leftImg8bit" / split / "aachen"
        ann_dir = tmp_path / "gtFine" / split / "aachen"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / "aachen_000000_000019_leftImg8bit.png"))

        instance_map = np.zeros((64, 64), dtype=np.uint16)
        instance_map[5:25, 5:25] = 26001   # car (class 26)
        instance_map[30:55, 5:55] = 31001  # train (class 31)
        inst_img = Image.fromarray(instance_map)
        inst_img.save(str(ann_dir / "aachen_000000_000019_gtFine_instanceIds.png"))

        return str(tmp_path)

    def test_label_map_override_excludes_train(self, tmp_path):
        """CITYSCAPES_BDD100K_LABEL_MAP must cause train instances to be ignored."""
        from src.datasets.cityscapes import (
            CITYSCAPES_BDD100K_LABEL_MAP,
            CityscapesDetectionDataset,
        )

        root = self._make_cityscapes_root_with_train_instance(tmp_path)
        ds = CityscapesDetectionDataset(
            root, split="train", label_map=CITYSCAPES_BDD100K_LABEL_MAP
        )
        _, target = ds[0]
        # Only the car box should survive; train is not in the 7-class map
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == CITYSCAPES_BDD100K_LABEL_MAP[26]


class TestFoggyCityscapesDataset:
    def _make_foggy_root(self, tmp_path, split="val", beta=0.02):
        """Create a minimal Foggy Cityscapes directory structure."""
        suffix = f"_leftImg8bit_foggy_beta_{beta:.2f}.png"
        img_dir = tmp_path / "leftImg8bit_foggy" / split / "aachen"
        ann_dir = tmp_path / "gtFine" / split / "aachen"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        stem = "aachen_000000_000019"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / (stem + suffix)))

        # Instance map with a single car instance
        instance_map = np.zeros((64, 64), dtype=np.uint16)
        instance_map[5:35, 5:35] = 26001  # class 26 = car
        inst_img = Image.fromarray(instance_map)
        inst_img.save(str(ann_dir / (stem + "_gtFine_instanceIds.png")))

        return str(tmp_path)

    def test_len(self, tmp_path):
        root = self._make_foggy_root(tmp_path)
        ds = FoggyCityscapesDataset(root, split="val", beta=0.02)
        assert len(ds) == 1

    def test_getitem_target_keys(self, tmp_path):
        root = self._make_foggy_root(tmp_path)
        ds = FoggyCityscapesDataset(root, split="val", beta=0.02)
        image, target = ds[0]
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target

    def test_car_annotation_loaded(self, tmp_path):
        root = self._make_foggy_root(tmp_path)
        ds = FoggyCityscapesDataset(root, split="val", beta=0.02)
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1
        assert target["labels"][0].item() == CITYSCAPES_LABEL_MAP[26]

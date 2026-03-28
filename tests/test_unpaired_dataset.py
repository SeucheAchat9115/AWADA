"""Tests for UnpairedImageDataset covering basic functionality and directory validation."""

import numpy as np
import pytest
from PIL import Image

from awada.datasets.unpaired_dataset import UnpairedImageDataset


@pytest.fixture()
def unpaired_dirs(tmp_path):
    """Create a minimal directory structure: 2 domain-A and 3 domain-B images."""
    dir_a = tmp_path / "domain_A"
    dir_b = tmp_path / "domain_B"
    dir_a.mkdir()
    dir_b.mkdir()

    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(dir_a / f"a_{i:03d}.png"))

    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(dir_b / f"b_{i:03d}.png"))

    return str(dir_a), str(dir_b)


class TestUnpairedImageDataset:
    def test_len_equals_max_domain_size(self, unpaired_dirs):
        dir_a, dir_b = unpaired_dirs
        ds = UnpairedImageDataset(dir_a, dir_b, patch_size=32)
        assert len(ds) == 3  # max(2, 3)

    def test_getitem_returns_two_tensors(self, unpaired_dirs):
        dir_a, dir_b = unpaired_dirs
        ds = UnpairedImageDataset(dir_a, dir_b, patch_size=32)
        item = ds[0]
        assert len(item) == 2

    def test_tensor_shapes(self, unpaired_dirs):
        dir_a, dir_b = unpaired_dirs
        ds = UnpairedImageDataset(dir_a, dir_b, patch_size=32)
        img_a, img_b = ds[0]
        assert img_a.shape == (3, 32, 32)
        assert img_b.shape == (3, 32, 32)

    def test_images_normalized(self, unpaired_dirs):
        dir_a, dir_b = unpaired_dirs
        ds = UnpairedImageDataset(dir_a, dir_b, patch_size=32)
        img_a, img_b = ds[0]
        assert img_a.min() >= -1.0 - 1e-5
        assert img_a.max() <= 1.0 + 1e-5
        assert img_b.min() >= -1.0 - 1e-5
        assert img_b.max() <= 1.0 + 1e-5


class TestUnpairedImageDatasetDirectoryChecks:
    """Tests that verify FileNotFoundError is raised for missing directories."""

    def test_missing_dir_a_raises(self, tmp_path):
        """FileNotFoundError must be raised when dir_A does not exist."""
        dir_b = tmp_path / "domain_B"
        dir_b.mkdir()
        with pytest.raises(FileNotFoundError, match="Domain A"):
            UnpairedImageDataset(str(tmp_path / "nonexistent_A"), str(dir_b))

    def test_missing_dir_b_raises(self, tmp_path):
        """FileNotFoundError must be raised when dir_B does not exist."""
        dir_a = tmp_path / "domain_A"
        dir_a.mkdir()
        with pytest.raises(FileNotFoundError, match="Domain B"):
            UnpairedImageDataset(str(dir_a), str(tmp_path / "nonexistent_B"))

    def test_both_dirs_present_does_not_raise(self, tmp_path):
        """No error should be raised when both directories exist."""
        dir_a = tmp_path / "domain_A"
        dir_b = tmp_path / "domain_B"
        dir_a.mkdir()
        dir_b.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(dir_a / "img.png"))
        img.save(str(dir_b / "img.png"))
        # Should not raise
        UnpairedImageDataset(str(dir_a), str(dir_b), patch_size=32)

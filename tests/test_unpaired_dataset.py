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
        """FileNotFoundError must be raised when source directory does not exist."""
        dir_b = tmp_path / "domain_B"
        dir_b.mkdir()
        with pytest.raises(FileNotFoundError, match="[Ss]ource"):
            UnpairedImageDataset(str(tmp_path / "nonexistent_A"), str(dir_b))

    def test_missing_dir_b_raises(self, tmp_path):
        """FileNotFoundError must be raised when target directory does not exist."""
        dir_a = tmp_path / "domain_A"
        dir_a.mkdir()
        with pytest.raises(FileNotFoundError, match="[Tt]arget"):
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


class TestUnpairedImageDatasetRecursiveSearch:
    """Tests that verify images are discovered recursively in sub-directories."""

    def test_images_in_subdirs_are_found(self, tmp_path):
        """Images nested inside sub-folders must be discovered."""
        dir_a = tmp_path / "domain_A"
        dir_b = tmp_path / "domain_B"
        subdir_a = dir_a / "subdir"
        subdir_b = dir_b / "subdir"
        subdir_a.mkdir(parents=True)
        subdir_b.mkdir(parents=True)

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        # Place images only in sub-directories (not at root level)
        img.save(str(subdir_a / "a_0.png"))
        img.save(str(subdir_a / "a_1.png"))
        img.save(str(subdir_b / "b_0.png"))

        ds = UnpairedImageDataset(str(dir_a), str(dir_b), patch_size=32)
        assert len(ds) == 2  # max(2, 1)

    def test_mixed_flat_and_nested_images_are_counted(self, tmp_path):
        """Images at root level and in sub-folders are all included in the count."""
        dir_a = tmp_path / "domain_A"
        dir_b = tmp_path / "domain_B"
        subdir_a = dir_a / "sub"
        dir_a.mkdir()
        dir_b.mkdir()
        subdir_a.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(dir_a / "root.png"))  # flat
        img.save(str(subdir_a / "nested.png"))  # nested
        img.save(str(dir_b / "b.png"))

        ds = UnpairedImageDataset(str(dir_a), str(dir_b), patch_size=32)
        # domain_A has 2 images (1 flat + 1 nested), domain_B has 1
        assert len(ds) == 2  # max(2, 1)

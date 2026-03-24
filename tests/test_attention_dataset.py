"""Tests for AttentionPairedDataset covering patch extraction, normalization, and attention map fallback behavior."""
import numpy as np
import pytest
from PIL import Image

from src.datasets.attention_dataset import AttentionPairedDataset


@pytest.fixture()
def dataset_dirs(tmp_path):
    """Create a minimal directory structure: 3 source and 2 target images."""
    src_dir = tmp_path / 'source'
    tgt_dir = tmp_path / 'target'
    att_dir = tmp_path / 'attention'
    src_dir.mkdir()
    tgt_dir.mkdir()
    att_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(str(src_dir / f'img_{i:03d}.png'))
        npy = np.zeros((64, 64), dtype=np.float32)
        npy[10:40, 10:40] = 1.0
        np.save(str(att_dir / f'img_{i:03d}.npy'), npy)

    for i in range(2):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(str(tgt_dir / f'tgt_{i:03d}.png'))

    return str(src_dir), str(tgt_dir), str(att_dir)


class TestAttentionPairedDataset:
    def test_len_equals_source_files(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att)
        assert len(ds) == 3

    def test_getitem_returns_four_tensors(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        item = ds[0]
        assert len(item) == 4

    def test_source_patch_shape(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        src_patch, tgt_patch, att_A, att_B = ds[0]
        assert src_patch.shape == (3, 32, 32)

    def test_target_patch_shape(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        src_patch, tgt_patch, att_A, att_B = ds[0]
        assert tgt_patch.shape == (3, 32, 32)

    def test_attention_A_shape(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        _, _, att_A, _ = ds[0]
        assert att_A.shape == (1, 32, 32)

    def test_attention_B_all_ones_when_no_target_att(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, target_attention_root=None, patch_size=32)
        _, _, _, att_B = ds[0]
        import torch
        assert (att_B == torch.ones(1, 32, 32)).all()

    def test_image_normalized_to_minus_one_one(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        src_patch, tgt_patch, _, _ = ds[0]
        assert src_patch.min() >= -1.0 - 1e-5
        assert src_patch.max() <= 1.0 + 1e-5

    def test_target_image_normalized(self, dataset_dirs):
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        _, tgt_patch, _, _ = ds[0]
        assert tgt_patch.min() >= -1.0 - 1e-5
        assert tgt_patch.max() <= 1.0 + 1e-5

    def test_attention_A_binary_values(self, dataset_dirs):
        """Source attention map should only contain 0 or 1."""
        src, tgt, att = dataset_dirs
        ds = AttentionPairedDataset(src, tgt, att, patch_size=32)
        import torch
        for i in range(len(ds)):
            _, _, att_A, _ = ds[i]
            unique = torch.unique(att_A)
            assert all(v.item() in (0.0, 1.0) for v in unique)

    def test_with_target_attention(self, tmp_path):
        """Dataset loads target attention maps when target_attention_root is provided."""
        src_dir = tmp_path / 'src'
        tgt_dir = tmp_path / 'tgt'
        att_dir = tmp_path / 'att'
        tgt_att_dir = tmp_path / 'tgt_att'
        for d in (src_dir, tgt_dir, att_dir, tgt_att_dir):
            d.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(src_dir / 'img_000.png'))
        np.save(str(att_dir / 'img_000.npy'), np.zeros((64, 64), dtype=np.float32))

        tgt_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        tgt_img.save(str(tgt_dir / 'tgt_000.png'))
        np.save(str(tgt_att_dir / 'tgt_000.npy'),
                np.ones((64, 64), dtype=np.float32))

        ds = AttentionPairedDataset(
            str(src_dir), str(tgt_dir), str(att_dir),
            target_attention_root=str(tgt_att_dir), patch_size=32
        )
        _, _, _, att_B = ds[0]
        assert att_B.shape == (1, 32, 32)

    def test_missing_attention_fallback_to_ones(self, tmp_path):
        """When attention .npy file is missing, fallback to all-ones mask."""
        src_dir = tmp_path / 'src'
        tgt_dir = tmp_path / 'tgt'
        att_dir = tmp_path / 'att'  # empty – no .npy files
        for d in (src_dir, tgt_dir, att_dir):
            d.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(src_dir / 'img_000.png'))

        tgt_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        tgt_img.save(str(tgt_dir / 'tgt_000.png'))

        ds = AttentionPairedDataset(str(src_dir), str(tgt_dir), str(att_dir), patch_size=32)
        _, _, att_A, _ = ds[0]
        import torch
        assert (att_A == torch.ones(1, 32, 32)).all()

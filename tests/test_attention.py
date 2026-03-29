"""Tests for generate_attention_maps function."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from awada.utils.attention import generate_attention_maps


class _FakeRPN(nn.Module):
    """Minimal RPN stand-in whose forward returns (boxes_list, scores_list).

    This allows ``register_forward_hook`` to fire just as it would on a real
    ``torch.nn.Module``, so the hook-based implementation in
    ``generate_attention_maps`` is exercised end-to-end.
    """

    def __init__(self, boxes_fn, scores_fn=None):
        super().__init__()
        self._boxes_fn = boxes_fn
        self._scores_fn = scores_fn

    def forward(self, imgs):
        boxes = self._boxes_fn(imgs)
        if self._scores_fn is not None:
            scores = self._scores_fn(imgs)
        else:
            scores = [torch.ones(len(b)) for b in boxes]
        return boxes, scores


def _build_detector(boxes_fn, scores_fn=None):
    """Return a mock detector whose RPN is a real nn.Module.

    boxes_fn(imgs) -> list of [N, 4] tensors (one per image).
    scores_fn(imgs) -> list of [N] tensors; defaults to all-ones when omitted.
    """
    detector = MagicMock()
    detector.eval.return_value = detector
    detector.to.return_value = detector

    fake_rpn = _FakeRPN(boxes_fn, scores_fn)
    detector.rpn = fake_rpn

    def forward(imgs):
        # Invoke the RPN module so that any registered forward hook fires.
        fake_rpn(imgs)

    detector.side_effect = forward
    return detector


def _simple_loader(n_images, H=32, W=32, batch_size=1):
    images = torch.randn(n_images, 3, H, W)
    dataset = TensorDataset(images)

    def collate(batch):
        return [b[0] for b in batch]

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


class TestGenerateAttentionMaps:
    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it does not exist."""
        output_dir = str(tmp_path / "attention_out")
        loader = _simple_loader(1)

        def boxes_fn(imgs):
            return [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")
        assert os.path.isdir(output_dir)

    def test_saves_npy_files(self, tmp_path):
        """One .npy file is saved for each image in the dataset."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(3)

        def boxes_fn(imgs):
            return [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
        assert len(npy_files) == 3

    def test_attention_map_shape(self, tmp_path):
        """Each saved attention map has the same H×W as the input image."""
        output_dir = str(tmp_path / "maps")
        H, W = 32, 48
        loader = _simple_loader(1, H=H, W=W)

        def boxes_fn(imgs):
            return [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.shape == (H, W)

    def test_attention_map_binary_values(self, tmp_path):
        """Saved attention maps should only contain 0.0 or 1.0."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1)

        def boxes_fn(imgs):
            return [torch.tensor([[2.0, 2.0, 20.0, 20.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        unique_vals = set(np.unique(arr).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_score_threshold_filters_boxes(self, tmp_path):
        """Only boxes with objectness score >= score_threshold are used."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1, H=64, W=64)

        # Two non-overlapping boxes: top-left (high score) and bottom-right (low score)
        all_boxes = torch.tensor(
            [
                [0.0, 0.0, 32.0, 32.0],  # high score – should be included
                [32.0, 32.0, 64.0, 64.0],  # low score  – should be excluded
            ]
        )
        all_scores = torch.tensor([0.8, 0.2])

        def boxes_fn(imgs):
            return [all_boxes for _ in imgs]

        def scores_fn(imgs):
            return [all_scores for _ in imgs]

        detector = _build_detector(boxes_fn, scores_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.shape == (64, 64)
        # Top-left quadrant (high-score box) should be foreground
        assert arr[:32, :32].sum() > 0
        # Bottom-right quadrant (low-score box) should remain background
        assert arr[32:, 32:].sum() == pytest.approx(0.0)

    def test_all_boxes_above_threshold_included(self, tmp_path):
        """All proposals with score >= threshold are included, not just top-k."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1, H=64, W=64)

        # Four non-overlapping boxes, all with score 0.9 (above threshold 0.5)
        all_boxes = torch.tensor(
            [
                [0.0, 0.0, 16.0, 16.0],
                [16.0, 0.0, 32.0, 16.0],
                [32.0, 0.0, 48.0, 16.0],
                [48.0, 0.0, 64.0, 16.0],
            ]
        )
        all_scores = torch.tensor([0.9, 0.9, 0.9, 0.9])

        def boxes_fn(imgs):
            return [all_boxes for _ in imgs]

        def scores_fn(imgs):
            return [all_scores for _ in imgs]

        detector = _build_detector(boxes_fn, scores_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        # All four boxes span the top 16 rows; verify the entire strip is foreground
        assert arr[0:16, :].sum() == pytest.approx(4 * 16 * 16)

    def test_no_proposals_produces_zero_map(self, tmp_path):
        """When RPN returns no boxes, the attention map should be all zeros."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1)

        def boxes_fn(imgs):
            return [torch.zeros(0, 4) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.sum() == pytest.approx(0.0)

    def test_saved_dtype_is_uint8(self, tmp_path):
        """Attention maps must be saved as np.uint8 to reduce disk usage."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1)

        def boxes_fn(imgs):
            return [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, score_threshold=0.5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.dtype == np.uint8

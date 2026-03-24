"""Tests for generate_attention_maps function."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.attention import generate_attention_maps


def _build_detector(boxes_fn, scores_fn=None):
    """Return a mock detector that provides proposals via a filter_proposals mock.

    boxes_fn(imgs) -> list of [N, 4] tensors (one per image).
    scores_fn(imgs) -> list of [N] tensors; defaults to all-ones when omitted.
    """
    detector = MagicMock()
    detector.eval.return_value = detector
    detector.to.return_value = detector

    # Shared state so forward() can pass images to filter_proposals
    _state = {"imgs": None}

    def filter_proposals_impl(*args, **kwargs):
        imgs = _state["imgs"]
        boxes_list = boxes_fn(imgs)
        if scores_fn is not None:
            scores_list = scores_fn(imgs)
        else:
            scores_list = [torch.ones(len(b)) for b in boxes_list]
        return boxes_list, scores_list

    detector.rpn = MagicMock()
    detector.rpn.filter_proposals = filter_proposals_impl

    def forward(imgs):
        _state["imgs"] = imgs
        # Call the (potentially monkey-patched) filter_proposals so that
        # generate_attention_maps can capture boxes and scores.
        detector.rpn.filter_proposals(None, None, None, None)

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
        all_boxes = torch.tensor([
            [0.0, 0.0, 32.0, 32.0],   # high score – should be included
            [32.0, 32.0, 64.0, 64.0],  # low score  – should be excluded
        ])
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
        all_boxes = torch.tensor([
            [0.0, 0.0, 16.0, 16.0],
            [16.0, 0.0, 32.0, 16.0],
            [32.0, 0.0, 48.0, 16.0],
            [48.0, 0.0, 64.0, 16.0],
        ])
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

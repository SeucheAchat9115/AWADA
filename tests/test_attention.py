"""Tests for generate_attention_maps function."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.attention import generate_attention_maps


def _build_detector(boxes_fn):
    """Return a mock detector that fires the RPN hook with boxes from boxes_fn(imgs)."""
    detector = MagicMock()
    detector.eval.return_value = detector
    detector.to.return_value = detector

    hook_store = {}

    def register_hook(fn):
        hook_store["fn"] = fn
        handle = MagicMock()
        handle.remove = MagicMock()
        return handle

    detector.rpn = MagicMock()
    detector.rpn.register_forward_hook.side_effect = register_hook

    def forward(imgs):
        boxes_list = boxes_fn(imgs)
        if "fn" in hook_store:
            hook_store["fn"](detector.rpn, None, (boxes_list, None))

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
        generate_attention_maps(detector, loader, output_dir, top_k=5, device="cpu")
        assert os.path.isdir(output_dir)

    def test_saves_npy_files(self, tmp_path):
        """One .npy file is saved for each image in the dataset."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(3)

        def boxes_fn(imgs):
            return [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, top_k=5, device="cpu")

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
        generate_attention_maps(detector, loader, output_dir, top_k=5, device="cpu")

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
        generate_attention_maps(detector, loader, output_dir, top_k=5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        unique_vals = set(np.unique(arr).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_top_k_limits_boxes(self, tmp_path):
        """Only top_k boxes are used per image."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1, H=64, W=64)

        many_boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0]] * 20)

        def boxes_fn(imgs):
            return [many_boxes for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, top_k=1, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.shape == (64, 64)
        assert arr.sum() > 0

    def test_no_proposals_produces_zero_map(self, tmp_path):
        """When RPN returns no boxes, the attention map should be all zeros."""
        output_dir = str(tmp_path / "maps")
        loader = _simple_loader(1)

        def boxes_fn(imgs):
            return [torch.zeros(0, 4) for _ in imgs]

        detector = _build_detector(boxes_fn)
        generate_attention_maps(detector, loader, output_dir, top_k=5, device="cpu")

        npy_files = sorted(os.listdir(output_dir))
        arr = np.load(os.path.join(output_dir, npy_files[0]))
        assert arr.sum() == pytest.approx(0.0)

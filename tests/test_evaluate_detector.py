"""Tests for evaluate_detector.py helper functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from evaluate_detector import build_model, collate_fn, evaluate, get_dataset, load_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader(n_images=2, H=32, W=32):
    """Return a DataLoader that yields (image_list, target_list) batches."""
    images = torch.randn(n_images, 3, H, W)
    targets = [
        {
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
        }
        for _ in range(n_images)
    ]
    dataset = list(zip(images, targets))
    return DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_model(self):
        model = build_model(num_classes=1)
        assert model is not None

    def test_box_predictor_output_size(self):
        """Box predictor should have num_classes + 1 outputs (background included)."""
        for nc in (1, 8):
            model = build_model(num_classes=nc)
            out_channels = model.roi_heads.box_predictor.cls_score.out_features
            assert out_channels == nc + 1

    def test_model_is_in_eval_mode_after_eval_call(self):
        model = build_model(num_classes=1)
        model.eval()
        assert not model.training


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_returns_tuple_of_tuples(self):
        images = [torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        targets = [{"labels": torch.tensor([1])}, {"labels": torch.tensor([2])}]
        batch = list(zip(images, targets))
        imgs, tgts = collate_fn(batch)
        assert isinstance(imgs, tuple)
        assert isinstance(tgts, tuple)
        assert len(imgs) == 2
        assert len(tgts) == 2


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------


class TestGetDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("nonexistent", "/fake/root", "val")

    @patch("evaluate_detector.CityscapesDetectionDataset")
    def test_cityscapes_dataset_instantiated(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_dataset("cityscapes", "/data/cs", "val", classes=["car"])
        mock_cls.assert_called_once_with("/data/cs", split="val", classes=["car"], transforms=None)

    @patch("evaluate_detector.FoggyCityscapesDataset")
    def test_foggy_cityscapes_dataset_instantiated(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_dataset("foggy_cityscapes", "/data/foggy", "val")
        mock_cls.assert_called_once_with("/data/foggy", split="val", transforms=None)

    @patch("evaluate_detector.Bdd100kDataset")
    def test_bdd100k_dataset_instantiated(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_dataset("bdd100k", "/data/bdd", "val")
        mock_cls.assert_called_once_with("/data/bdd", split="val", transforms=None)

    @patch("evaluate_detector.CityscapesDetectionDataset")
    def test_transforms_forwarded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from src.utils.transforms import ResizeToMinSize

        t = ResizeToMinSize(600)
        get_dataset("cityscapes", "/data/cs", "val", transforms=t)
        mock_cls.assert_called_once_with("/data/cs", split="val", classes=None, transforms=t)


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_loads_plain_state_dict(self, tmp_path):
        model = build_model(num_classes=1)
        ckpt_path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), ckpt_path)

        fresh_model = build_model(num_classes=1)
        loaded = load_checkpoint(fresh_model, ckpt_path, torch.device("cpu"))
        assert loaded is fresh_model

    def test_loads_checkpoint_dict_with_model_state_dict_key(self, tmp_path):
        model = build_model(num_classes=1)
        ckpt_path = str(tmp_path / "ckpt.pth")
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
            },
            ckpt_path,
        )

        fresh_model = build_model(num_classes=1)
        loaded = load_checkpoint(fresh_model, ckpt_path, torch.device("cpu"))
        assert loaded is fresh_model


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_dict_with_map_keys(self):
        """evaluate() must return a dict containing mAP@0.5 and mAP@0.5:0.95."""
        model = build_model(num_classes=1)
        loader = _make_loader(n_images=1)
        metrics = evaluate(model, loader, torch.device("cpu"), num_classes=1)
        assert "mAP@0.5" in metrics
        assert "mAP@0.5:0.95" in metrics

    def test_map_values_are_floats_in_range(self):
        model = build_model(num_classes=1)
        loader = _make_loader(n_images=1)
        metrics = evaluate(model, loader, torch.device("cpu"), num_classes=1)
        for key in ("mAP@0.5", "mAP@0.5:0.95"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_map50_ge_map50_95(self):
        """mAP@0.5 must be >= mAP@0.5:0.95 (looser threshold yields higher/equal metric)."""
        model = build_model(num_classes=1)
        loader = _make_loader(n_images=2)
        metrics = evaluate(model, loader, torch.device("cpu"), num_classes=1)
        assert metrics["mAP@0.5"] >= metrics["mAP@0.5:0.95"] - 1e-6


# ---------------------------------------------------------------------------
# main() — CLI integration via subprocess
# ---------------------------------------------------------------------------


class TestMain:
    def test_writes_results_txt(self, tmp_path):
        """Running main() end-to-end should create results.txt in output_dir."""
        model = build_model(num_classes=1)
        ckpt_path = str(tmp_path / "det.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Patch the dataset so we don't need real data on disk
        mock_dataset = [
            (
                torch.randn(3, 64, 64),
                {
                    "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0]]),
                    "labels": torch.tensor([1]),
                },
            )
        ]

        with patch("evaluate_detector.get_dataset", return_value=mock_dataset):
            from evaluate_detector import main

            with patch(
                "sys.argv",
                [
                    "evaluate_detector.py",
                    "--detector_checkpoint",
                    ckpt_path,
                    "--dataset",
                    "cityscapes",
                    "--data_root",
                    "/fake",
                    "--num_classes",
                    "1",
                    "--output_dir",
                    str(tmp_path),
                    "--device",
                    "cpu",
                    "--label",
                    "Test Experiment",
                    "--benchmark",
                    "sim10k_to_cityscapes",
                ],
            ):
                main()

        results_path = tmp_path / "results.txt"
        assert results_path.exists()
        content = results_path.read_text()
        assert "Test Experiment" in content
        assert "sim10k_to_cityscapes" in content
        assert "mAP@0.5:" in content
        assert "mAP@0.5:0.95:" in content

    def test_results_txt_without_label_or_benchmark(self, tmp_path):
        """results.txt should still be written even when --label and --benchmark are omitted."""
        model = build_model(num_classes=8)
        ckpt_path = str(tmp_path / "det.pth")
        torch.save(model.state_dict(), ckpt_path)

        mock_dataset = [
            (
                torch.randn(3, 64, 64),
                {
                    "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0]]),
                    "labels": torch.tensor([1]),
                },
            )
        ]

        with patch("evaluate_detector.get_dataset", return_value=mock_dataset):
            from evaluate_detector import main

            with patch(
                "sys.argv",
                [
                    "evaluate_detector.py",
                    "--detector_checkpoint",
                    ckpt_path,
                    "--dataset",
                    "foggy_cityscapes",
                    "--data_root",
                    "/fake",
                    "--num_classes",
                    "8",
                    "--output_dir",
                    str(tmp_path),
                    "--device",
                    "cpu",
                ],
            ):
                main()

        results_path = tmp_path / "results.txt"
        assert results_path.exists()
        content = results_path.read_text()
        assert "mAP@0.5:" in content

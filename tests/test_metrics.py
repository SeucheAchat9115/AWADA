"""Tests for compute_map_range and _to_coco_format."""

import pytest
import torch

from awada.utils.metrics import _to_coco_format, compute_map_range


def _make_perfect_pair(n_boxes=3, num_classes=2):
    """Return a (predictions, targets) pair where predicted boxes == ground-truth boxes."""
    boxes = torch.tensor(
        [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0], [5.0, 5.0, 30.0, 30.0]],
        dtype=torch.float32,
    )[:n_boxes]
    labels = torch.tensor([1, 2, 1], dtype=torch.long)[:n_boxes]
    scores = torch.ones(n_boxes)

    target = {"boxes": boxes, "labels": labels}
    pred = {"boxes": boxes.clone(), "labels": labels.clone(), "scores": scores}
    return [pred], [target]


class TestToCocoFormat:
    def test_returns_gt_dataset_and_results(self):
        preds, targets = _make_perfect_pair()
        gt_dataset, results = _to_coco_format(preds, targets, num_classes=2)
        assert "images" in gt_dataset
        assert "annotations" in gt_dataset
        assert "categories" in gt_dataset
        assert isinstance(results, list)

    def test_correct_number_of_annotations(self):
        preds, targets = _make_perfect_pair(n_boxes=3)
        gt_dataset, _ = _to_coco_format(preds, targets, num_classes=2)
        assert len(gt_dataset["annotations"]) == 3

    def test_correct_number_of_results(self):
        preds, targets = _make_perfect_pair(n_boxes=3)
        _, results = _to_coco_format(preds, targets, num_classes=2)
        assert len(results) == 3

    def test_category_ids_are_one_indexed(self):
        preds, targets = _make_perfect_pair()
        gt_dataset, _ = _to_coco_format(preds, targets, num_classes=2)
        cat_ids = {c["id"] for c in gt_dataset["categories"]}
        assert 0 not in cat_ids
        assert 1 in cat_ids

    def test_out_of_range_labels_skipped(self):
        """Labels outside [1, num_classes] should be silently ignored."""
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        labels = torch.tensor([99])  # way out of range for num_classes=2
        scores = torch.tensor([0.9])
        preds = [{"boxes": boxes, "labels": labels, "scores": scores}]
        targets = [{"boxes": boxes, "labels": labels}]
        _, results = _to_coco_format(preds, targets, num_classes=2)
        assert len(results) == 0

    def test_empty_predictions(self):
        preds = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "scores": torch.zeros(0),
            }
        ]
        targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
        _, results = _to_coco_format(preds, targets, num_classes=1)
        assert results == []

    def test_bbox_format_is_xywh(self):
        """COCO format uses [x, y, w, h], not [x1, y1, x2, y2]."""
        boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
        labels = torch.tensor([1])
        scores = torch.tensor([1.0])
        preds = [{"boxes": boxes, "labels": labels, "scores": scores}]
        targets = [{"boxes": boxes, "labels": labels}]
        gt_dataset, results = _to_coco_format(preds, targets, num_classes=1)

        ann_bbox = gt_dataset["annotations"][0]["bbox"]
        x, y, w, h = ann_bbox
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)
        assert w == pytest.approx(40.0)  # 50 - 10
        assert h == pytest.approx(60.0)  # 80 - 20


class TestComputeMapRange:
    def test_returns_dict_with_expected_keys(self):
        preds, targets = _make_perfect_pair()
        result = compute_map_range(preds, targets, num_classes=2)
        assert "mAP@0.5" in result
        assert "mAP@0.5:0.95" in result

    def test_empty_predictions_returns_zeros(self):
        preds = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "scores": torch.zeros(0),
            }
        ]
        targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])}]
        result = compute_map_range(preds, targets, num_classes=1)
        assert result["mAP@0.5"] == pytest.approx(0.0)
        assert result["mAP@0.5:0.95"] == pytest.approx(0.0)

    def test_perfect_prediction_high_map(self):
        """Perfect predictions (boxes and labels exactly matching) should yield mAP = 1."""
        preds, targets = _make_perfect_pair(n_boxes=2, num_classes=2)
        result = compute_map_range(preds, targets, num_classes=2)
        assert result["mAP@0.5"] == pytest.approx(1.0, abs=0.01)

    def test_map_50_ge_map_50_95(self):
        """mAP@0.5 is always >= mAP@0.5:0.95 (stricter threshold)."""
        preds, targets = _make_perfect_pair()
        result = compute_map_range(preds, targets, num_classes=2)
        assert result["mAP@0.5"] >= result["mAP@0.5:0.95"] - 1e-6

    def test_num_classes_inferred_from_data(self):
        """num_classes can be inferred automatically from target labels."""
        preds, targets = _make_perfect_pair()
        result = compute_map_range(preds, targets, num_classes=None)
        assert "mAP@0.5" in result

    def test_multiple_images(self):
        """compute_map_range handles a list of multiple image predictions."""
        preds, targets = _make_perfect_pair(n_boxes=2)
        # Duplicate to simulate multiple images
        preds = preds * 3
        targets = targets * 3
        result = compute_map_range(preds, targets, num_classes=2)
        assert result["mAP@0.5"] > 0.0

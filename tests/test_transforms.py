"""Tests for the ResizeToMinSize transform."""

import torch

from src.utils.transforms import ResizeToMinSize


class TestResizeToMinSizeInit:
    def test_default_min_size(self):
        t = ResizeToMinSize()
        assert t.min_size == 600

    def test_custom_min_size(self):
        t = ResizeToMinSize(min_size=300)
        assert t.min_size == 300

    def test_invalid_min_size_raises(self):
        import pytest

        with pytest.raises(ValueError):
            ResizeToMinSize(min_size=0)

        with pytest.raises(ValueError):
            ResizeToMinSize(min_size=-1)


class TestResizeToMinSizeCall:
    def _make_sample(self, h, w, n_boxes=1):
        image = torch.zeros(3, h, w)
        if n_boxes > 0:
            boxes = torch.tensor([[5.0, 5.0, float(w // 2), float(h // 2)] for _ in range(n_boxes)])
            labels = torch.ones(n_boxes, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return image, target

    # ------------------------------------------------------------------
    # Image resizing
    # ------------------------------------------------------------------

    def test_portrait_image_shortest_side_resized(self):
        """Portrait image: width (shorter) should become min_size."""
        h, w = 800, 600
        image, target = self._make_sample(h, w)
        t = ResizeToMinSize(min_size=300)
        out_img, _ = t(image, target)
        assert min(out_img.shape[1], out_img.shape[2]) == 300

    def test_landscape_image_shortest_side_resized(self):
        """Landscape image: height (shorter) should become min_size."""
        h, w = 600, 1000
        image, target = self._make_sample(h, w)
        t = ResizeToMinSize(min_size=300)
        out_img, _ = t(image, target)
        assert min(out_img.shape[1], out_img.shape[2]) == 300

    def test_square_image_both_sides_equal_min_size(self):
        h = w = 256
        image, target = self._make_sample(h, w)
        t = ResizeToMinSize(min_size=128)
        out_img, _ = t(image, target)
        assert out_img.shape[1] == 128
        assert out_img.shape[2] == 128

    def test_no_resize_when_already_min_size(self):
        """Image whose shortest side already equals min_size should be returned unchanged."""
        h, w = 600, 800
        image, target = self._make_sample(h, w)
        t = ResizeToMinSize(min_size=600)
        out_img, _ = t(image, target)
        assert out_img.shape[1] == 600
        assert out_img.shape[2] == 800

    def test_channel_count_preserved(self):
        h, w = 100, 150
        image = torch.zeros(3, h, w)
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
        t = ResizeToMinSize(min_size=50)
        out_img, _ = t(image, target)
        assert out_img.shape[0] == 3

    # ------------------------------------------------------------------
    # Bounding-box scaling
    # ------------------------------------------------------------------

    def test_boxes_scaled_proportionally(self):
        """Bounding-box coordinates should be multiplied by the same scale as the image."""
        h, w = 600, 800
        image, target = self._make_sample(h, w, n_boxes=1)
        original_box = target["boxes"].clone()
        t = ResizeToMinSize(min_size=300)
        _, out_target = t(image, target)
        scale = 300 / min(h, w)
        expected = original_box * scale
        assert torch.allclose(out_target["boxes"], expected, atol=1e-3)

    def test_empty_boxes_handled(self):
        """Images with no annotations should not raise an error."""
        h, w = 640, 480
        image, target = self._make_sample(h, w, n_boxes=0)
        t = ResizeToMinSize(min_size=320)
        out_img, out_target = t(image, target)
        assert out_target["boxes"].shape == (0, 4)

    def test_original_target_not_mutated(self):
        """The transform must not modify the original target dict in-place."""
        h, w = 640, 480
        image, target = self._make_sample(h, w, n_boxes=2)
        original_boxes = target["boxes"].clone()
        t = ResizeToMinSize(min_size=320)
        t(image, target)
        assert torch.allclose(target["boxes"], original_boxes)

    def test_non_box_target_keys_preserved(self):
        """Keys other than 'boxes' should pass through unchanged."""
        h, w = 640, 480
        image, target = self._make_sample(h, w)
        target["image_id"] = torch.tensor([42])
        t = ResizeToMinSize(min_size=320)
        _, out_target = t(image, target)
        assert "image_id" in out_target
        assert out_target["image_id"].item() == 42

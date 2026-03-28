"""Dataset transforms for object detection."""

import torch
import torchvision.transforms.functional as TF


class ResizeToMinSize:
    """Resize an image tensor so its shortest side equals ``min_size``.

    Bounding boxes in the accompanying target dict are scaled accordingly.

    Args:
        min_size: Target length (in pixels) for the shortest image side.
    """

    def __init__(self, min_size: int = 600):
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        self.min_size = min_size

    def __call__(
        self, image: torch.Tensor, target: dict | None
    ) -> tuple[torch.Tensor, dict | None]:
        _, h, w = image.shape
        scale = self.min_size / min(h, w)
        if scale == 1.0:
            return image, target
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image = TF.resize(image, [new_h, new_w])
        if target is not None and target.get("boxes") is not None:
            target = dict(target)  # shallow copy to avoid mutating the original
            target["boxes"] = target["boxes"].clone() * scale
        return image, target

"""Unpaired image dataset used for CycleGAN-style training."""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class UnpairedImageDataset(Dataset):
    """Dataset of unpaired images from two domains A and B.

    Images are discovered recursively inside the given directories (including
    all sub-folders) and a random crop + horizontal flip augmentation is applied
    on-the-fly.  When the two domains have different numbers of images the
    shorter list is cycled so that every epoch consumes the same number of
    iterations regardless of domain size.
    """

    def __init__(self, source: str, target: str, patch_size: int = 128) -> None:
        """Initialise the unpaired image dataset.

        Args:
            source: Root directory for source-domain images (searched recursively).
            target: Root directory for target-domain images (searched recursively).
            patch_size: Side length of the square random crops (default: 128).
        """
        if not os.path.isdir(source):
            raise FileNotFoundError(
                f"Source directory not found: '{source}'. "
                "Please ensure the source domain images are present before constructing the dataset."
            )
        if not os.path.isdir(target):
            raise FileNotFoundError(
                f"Target directory not found: '{target}'. "
                "Please ensure the target domain images are present before constructing the dataset."
            )
        self.patch_size = patch_size
        self.source_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(source)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.target_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(target)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.transform = T.Compose(
            [
                T.RandomCrop(patch_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        """Return the number of iterations per epoch (max of both domain sizes)."""
        return max(len(self.source_files), len(self.target_files))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a randomly augmented pair of source and target domain images.

        Args:
            idx: Sample index (cycled within each domain independently).

        Returns:
            Tuple of ``(source_image, target_image)`` tensors of shape
            ``[3, patch_size, patch_size]`` normalised to ``[-1, 1]``.
        """
        source_img = Image.open(self.source_files[idx % len(self.source_files)]).convert("RGB")
        target_img = Image.open(self.target_files[idx % len(self.target_files)]).convert("RGB")
        return self.transform(source_img), self.transform(target_img)

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

    def __init__(self, dir_A: str, dir_B: str, patch_size: int = 128) -> None:
        """Initialise the unpaired image dataset.

        Args:
            dir_A: Root directory for domain-A images (searched recursively).
            dir_B: Root directory for domain-B images (searched recursively).
            patch_size: Side length of the square random crops (default: 128).
        """
        if not os.path.isdir(dir_A):
            raise FileNotFoundError(
                f"Domain A directory not found: '{dir_A}'. "
                "Please ensure the domain A images are present before constructing the dataset."
            )
        if not os.path.isdir(dir_B):
            raise FileNotFoundError(
                f"Domain B directory not found: '{dir_B}'. "
                "Please ensure the domain B images are present before constructing the dataset."
            )
        self.patch_size = patch_size
        self.files_A = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(dir_A)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.files_B = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(dir_B)
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
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a randomly augmented pair of domain-A and domain-B images.

        Args:
            idx: Sample index (cycled within each domain independently).

        Returns:
            Tuple of ``(image_A, image_B)`` tensors of shape
            ``[3, patch_size, patch_size]`` normalised to ``[-1, 1]``.
        """
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[idx % len(self.files_B)]).convert("RGB")
        return self.transform(img_A), self.transform(img_B)

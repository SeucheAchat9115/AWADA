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

    def __init__(self, source_dir: str, target_dir: str, patch_size: int = 256) -> None:
        """Initialise the unpaired image dataset.

        Args:
            source_dir: Root directory for source-domain images (searched recursively).
            target_dir: Root directory for target-domain images (searched recursively).
            patch_size: Side length of the square random crops (default: 256).
                Follows the canonical CycleGAN preprocessing: images are first
                resized to ``(patch_size + 30) × (patch_size + 30)`` and then a
                random square crop of ``patch_size`` is taken, matching the
                standard 286→256 pipeline from Zhu et al. (2017).
        """
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(
                f"Source directory not found: '{source_dir}'. "
                "Please ensure the source domain images are present before constructing the dataset."
            )
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(
                f"Target directory not found: '{target_dir}'. "
                "Please ensure the target domain images are present before constructing the dataset."
            )
        self.patch_size = patch_size
        self.source_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(source_dir)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.target_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(target_dir)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        load_size = patch_size + 30
        self.transform = T.Compose(
            [
                T.Resize((load_size, load_size)),
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

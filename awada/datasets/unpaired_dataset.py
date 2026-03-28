"""Unpaired image dataset used for CycleGAN-style training."""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class UnpairedImageDataset(Dataset):
    """Dataset of unpaired images from two domains A and B.

    Images are loaded from flat directories and a random crop + horizontal flip
    augmentation is applied on-the-fly.  When the two domains have different
    numbers of images the shorter list is cycled so that every epoch consumes
    the same number of iterations regardless of domain size.
    """

    def __init__(self, dir_A: str, dir_B: str, patch_size: int = 128):
        self.patch_size = patch_size
        self.files_A = sorted(
            [
                os.path.join(dir_A, f)
                for f in os.listdir(dir_A)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.files_B = sorted(
            [
                os.path.join(dir_B, f)
                for f in os.listdir(dir_B)
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
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[idx % len(self.files_B)]).convert("RGB")
        return self.transform(img_A), self.transform(img_B)

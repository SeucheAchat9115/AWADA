from typing import cast

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator used in CycleGAN.

    Classifies overlapping image patches as real or fake rather than the
    whole image, which encourages high-frequency sharpness and is more
    parameter-efficient than a global discriminator.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        ndf: Base number of discriminator filters (default: 64).
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64) -> None:
        """Initialise the PatchGAN discriminator.

        Args:
            in_channels: Number of input image channels.
            ndf: Base number of discriminator filters.
        """
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the discriminator forward pass.

        Args:
            x: Input image tensor of shape ``[B, in_channels, H, W]``.

        Returns:
            Patch-level prediction map of shape ``[B, 1, H', W']``.
        """
        return cast(torch.Tensor, self.model(x))

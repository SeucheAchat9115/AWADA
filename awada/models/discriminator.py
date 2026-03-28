import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
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
        return self.model(x)

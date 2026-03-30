import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with instance normalisation used in the ResNet generator.

    Applies two 3×3 convolutions with reflection padding and skips the input
    via an additive shortcut connection, as in He et al. (2016).

    Args:
        channels: Number of input (and output) feature-map channels.
    """

    def __init__(self, channels: int) -> None:
        """Initialise the residual block.

        Args:
            channels: Number of input and output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input feature map of shape ``[B, channels, H, W]``.

        Returns:
            Output feature map of the same shape, with the residual added.
        """
        return x + self.block(x)  # type: ignore[no-any-return]


class ResNetGenerator(nn.Module):
    """ResNet-based image-to-image generator used in CycleGAN.

    Encodes the input with two stride-2 downsampling layers, processes it
    through ``n_blocks`` residual blocks, then decodes with two fractionally
    strided (transpose) convolutions.  Reflection padding avoids border
    artefacts.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        out_channels: Number of output image channels (default: 3 for RGB).
        ngf: Base number of generator filters (default: 64).
        n_blocks: Number of residual blocks in the bottleneck (default: 9).
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64, n_blocks: int = 9
    ) -> None:
        """Initialise the ResNet generator.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output image channels.
            ngf: Base number of generator filters.
            n_blocks: Number of residual blocks.
        """
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsampling
        for i in range(2):
            mult = 2**i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf * 4))
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the generator forward pass.

        Args:
            x: Input image tensor of shape ``[B, in_channels, H, W]``.

        Returns:
            Translated image tensor of shape ``[B, out_channels, H, W]``, with
            pixel values in ``[-1, 1]`` due to the final ``Tanh`` activation.
        """
        return self.model(x)  # type: ignore[no-any-return]

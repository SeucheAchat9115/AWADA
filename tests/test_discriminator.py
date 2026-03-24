"""Tests for PatchGANDiscriminator."""

import torch

from src.models.discriminator import PatchGANDiscriminator


class TestPatchGANDiscriminator:
    def test_default_output_shape(self):
        """Discriminator maps a 3-channel image to a single-channel patch map."""
        model = PatchGANDiscriminator()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1

    def test_batch_output_shape(self):
        """Batch dimension is preserved in discriminator output."""
        model = PatchGANDiscriminator()
        x = torch.randn(4, 3, 256, 256)
        out = model(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 1

    def test_custom_in_channels(self):
        """Discriminator accepts custom in_channels."""
        model = PatchGANDiscriminator(in_channels=1)
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1

    def test_custom_ndf(self):
        """Discriminator works with different ndf values."""
        model = PatchGANDiscriminator(ndf=32)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 1

    def test_output_not_nan(self):
        """Discriminator output should not contain NaN values."""
        model = PatchGANDiscriminator()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_output_spatial_size_smaller_than_input(self):
        """PatchGAN output spatial size is smaller than input due to strided convs."""
        model = PatchGANDiscriminator()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape[2] < 256
        assert out.shape[3] < 256

    def test_gradients_flow(self):
        """Gradients should flow back through discriminator parameters."""
        model = PatchGANDiscriminator()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

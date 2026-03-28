"""Tests for ResidualBlock and ResNetGenerator covering shape preservation, output ranges, and gradient flow."""

import torch

from awada.models.generator import ResidualBlock, ResNetGenerator


class TestResidualBlock:
    def test_output_shape_unchanged(self):
        """Residual block preserves spatial dimensions and channel count."""
        block = ResidualBlock(channels=64)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero-initialized block weights the output equals the input."""
        block = ResidualBlock(channels=8)
        # Zero out the block weights so the residual path returns exactly x
        for p in block.block.parameters():
            torch.nn.init.zeros_(p)
        x = torch.randn(1, 8, 16, 16)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_gradients_flow(self):
        """Gradients should flow through the residual block."""
        block = ResidualBlock(channels=16)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        out = block(x)
        out.mean().backward()
        assert x.grad is not None


class TestResNetGenerator:
    def test_default_output_shape(self):
        """Generator output has the same spatial size and out_channels as specified."""
        gen = ResNetGenerator()
        x = torch.randn(1, 3, 128, 128)
        out = gen(x)
        assert out.shape == (1, 3, 128, 128)

    def test_output_range(self):
        """Generator uses Tanh, so all outputs should be in [-1, 1]."""
        gen = ResNetGenerator()
        x = torch.randn(1, 3, 128, 128)
        out = gen(x)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_custom_channels(self):
        """Generator works with custom in/out channel counts."""
        gen = ResNetGenerator(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64)
        out = gen(x)
        assert out.shape == (1, 1, 64, 64)

    def test_fewer_residual_blocks(self):
        """Generator with n_blocks=6 still produces correct output shape."""
        gen = ResNetGenerator(n_blocks=6)
        x = torch.randn(1, 3, 128, 128)
        out = gen(x)
        assert out.shape == (1, 3, 128, 128)

    def test_batch_dimension(self):
        """Generator handles batch size > 1."""
        gen = ResNetGenerator()
        x = torch.randn(2, 3, 128, 128)
        out = gen(x)
        assert out.shape == (2, 3, 128, 128)

    def test_output_not_nan(self):
        """Generator output should not contain NaN values."""
        gen = ResNetGenerator()
        x = torch.randn(1, 3, 128, 128)
        out = gen(x)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        """Gradients should flow back through generator parameters."""
        gen = ResNetGenerator()
        x = torch.randn(1, 3, 64, 64)
        out = gen(x)
        out.mean().backward()
        for p in gen.parameters():
            assert p.grad is not None

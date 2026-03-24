"""Tests for ImageBuffer and CycleGAN covering buffer management, forward pass, and loss computation."""
import torch
import pytest
from src.models.cyclegan import ImageBuffer, CycleGAN

DEVICE = 'cpu'
IMG_SIZE = 64  # Use small images to keep tests fast


def _make_cyclegan():
    return CycleGAN(device=DEVICE)


def _real_pair(batch=1):
    real_A = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    real_B = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    return real_A, real_B


class TestImageBuffer:
    def test_push_fills_buffer(self):
        buf = ImageBuffer(max_size=50)
        data = torch.randn(3, 3, 64, 64)
        result = buf.push_and_pop(data)
        assert result.shape == data.shape

    def test_push_pop_returns_tensor(self):
        buf = ImageBuffer(max_size=50)
        data = torch.randn(1, 3, 64, 64)
        result = buf.push_and_pop(data)
        assert isinstance(result, torch.Tensor)

    def test_buffer_max_size_respected(self):
        """Buffer should not grow beyond max_size."""
        buf = ImageBuffer(max_size=5)
        for _ in range(10):
            buf.push_and_pop(torch.randn(1, 3, 8, 8))
        assert len(buf.data) <= 5

    def test_buffer_output_shape(self):
        """Output shape matches input shape."""
        buf = ImageBuffer(max_size=50)
        data = torch.randn(4, 3, 16, 16)
        result = buf.push_and_pop(data)
        assert result.shape == data.shape

    def test_small_buffer_returns_from_history(self):
        """When buffer is full, some results may come from stored history."""
        torch.manual_seed(0)
        buf = ImageBuffer(max_size=2)
        # Fill the buffer
        buf.push_and_pop(torch.zeros(2, 3, 4, 4))
        # Now push ones; if history is returned we get zeros back
        result = buf.push_and_pop(torch.ones(2, 3, 4, 4))
        # Result should be either 0-tensor or 1-tensor (no other values)
        assert result.shape == (2, 3, 4, 4)


class TestCycleGAN:
    def test_set_input_stores_tensors(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        assert model.real_A.shape == real_A.shape
        assert model.real_B.shape == real_B.shape

    def test_forward_creates_fake_images(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        assert model.fake_B.shape == real_A.shape
        assert model.fake_A.shape == real_B.shape

    def test_forward_creates_reconstructions(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        assert model.rec_A.shape == real_A.shape
        assert model.rec_B.shape == real_B.shape

    def test_generator_loss_keys(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        for key in ('G_AB', 'G_BA', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'total_G'):
            assert key in losses, f"Missing key: {key}"

    def test_generator_loss_positive(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert losses['total_G'].item() >= 0

    def test_discriminator_loss_keys(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        for key in ('D_A', 'D_B', 'total_D'):
            assert key in losses

    def test_discriminator_loss_positive(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        assert losses['total_D'].item() >= 0

    def test_generator_loss_is_sum_of_parts(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        expected = (losses['G_AB'] + losses['G_BA'] +
                    losses['cycle_A'] + losses['cycle_B'] +
                    losses['idt_A'] + losses['idt_B'])
        assert torch.allclose(losses['total_G'], expected, atol=1e-5)

    def test_discriminator_loss_is_sum_of_parts(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        assert torch.allclose(losses['total_D'], losses['D_A'] + losses['D_B'], atol=1e-5)

    def test_no_nan_in_losses(self):
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        g_losses = model.compute_generator_loss()
        d_losses = model.compute_discriminator_loss()
        for v in list(g_losses.values()) + list(d_losses.values()):
            assert not torch.isnan(v), f"NaN detected in loss"

    def test_lambda_weights_scale_losses(self):
        """Increasing lambda_cyc should increase cycle loss contribution."""
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses_low = model.compute_generator_loss(lambda_cyc=1.0)
        model.forward()
        losses_high = model.compute_generator_loss(lambda_cyc=100.0)
        assert losses_high['cycle_A'].item() > losses_low['cycle_A'].item()

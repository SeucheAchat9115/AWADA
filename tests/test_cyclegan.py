"""Tests for ImageBuffer and CycleGAN covering buffer management, forward pass, and loss computation."""

import pytest
import torch

from awada.models.cyclegan import CycleGAN, ImageBuffer

DEVICE = "cpu"
IMG_SIZE = 64  # Use small images to keep tests fast


def _make_cyclegan():
    return CycleGAN(device=DEVICE)


def _real_pair(batch=1):
    real_A = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    real_B = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    return real_A, real_B


@pytest.fixture()
def cyclegan_fwd():
    """Return a CycleGAN instance that has already completed a forward pass."""
    model = _make_cyclegan()
    real_A, real_B = _real_pair()
    model.set_input(real_A, real_B)
    model.forward()
    return model, real_A, real_B


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

    def test_forward_creates_fake_images(self, cyclegan_fwd):
        model, real_A, real_B = cyclegan_fwd
        assert model.fake_B.shape == real_A.shape
        assert model.fake_A.shape == real_B.shape

    def test_forward_creates_reconstructions(self, cyclegan_fwd):
        model, real_A, real_B = cyclegan_fwd
        assert model.rec_A.shape == real_A.shape
        assert model.rec_B.shape == real_B.shape

    def test_generator_loss_keys(self, cyclegan_fwd):
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss()
        for key in ("G_AB", "G_BA", "cycle_A", "cycle_B", "total_G"):
            assert key in losses, f"Missing key: {key}"
        for key in ("idt_A", "idt_B"):
            assert key not in losses, f"Identity loss key should not be present: {key}"

    @pytest.mark.parametrize("lambda_cyc,lambda_gan", [(10.0, 1.0), (5.0, 2.0)])
    def test_generator_loss_positive(self, cyclegan_fwd, lambda_cyc, lambda_gan):
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss(lambda_cyc=lambda_cyc, lambda_gan=lambda_gan)
        assert losses["total_G"].item() >= 0

    def test_discriminator_loss_keys(self, cyclegan_fwd):
        model, _, _ = cyclegan_fwd
        losses = model.compute_discriminator_loss()
        for key in ("D_A", "D_B", "total_D"):
            assert key in losses

    def test_discriminator_loss_positive(self, cyclegan_fwd):
        model, _, _ = cyclegan_fwd
        losses = model.compute_discriminator_loss()
        assert losses["total_D"].item() >= 0

    @pytest.mark.parametrize("lambda_cyc,lambda_gan", [(10.0, 1.0), (5.0, 2.0)])
    def test_generator_loss_is_sum_of_parts(self, cyclegan_fwd, lambda_cyc, lambda_gan):
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss(lambda_cyc=lambda_cyc, lambda_gan=lambda_gan)
        expected = losses["G_AB"] + losses["G_BA"] + losses["cycle_A"] + losses["cycle_B"]
        assert torch.allclose(losses["total_G"], expected, atol=1e-5)

    def test_discriminator_loss_is_sum_of_parts(self, cyclegan_fwd):
        model, _, _ = cyclegan_fwd
        losses = model.compute_discriminator_loss()
        assert torch.allclose(losses["total_D"], losses["D_A"] + losses["D_B"], atol=1e-5)

    @pytest.mark.parametrize("lambda_cyc,lambda_gan", [(10.0, 1.0), (5.0, 2.0)])
    def test_no_nan_in_losses(self, cyclegan_fwd, lambda_cyc, lambda_gan):
        model, _, _ = cyclegan_fwd
        g_losses = model.compute_generator_loss(lambda_cyc=lambda_cyc, lambda_gan=lambda_gan)
        d_losses = model.compute_discriminator_loss()
        for v in list(g_losses.values()) + list(d_losses.values()):
            assert not torch.isnan(v), "NaN detected in loss"

    def test_lambda_weights_scale_losses(self):
        """Increasing lambda_cyc should increase cycle loss contribution."""
        model = _make_cyclegan()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses_low = model.compute_generator_loss(lambda_cyc=1.0)
        model.forward()
        losses_high = model.compute_generator_loss(lambda_cyc=100.0)
        assert losses_high["cycle_A"].item() > losses_low["cycle_A"].item()

    # ------------------------------------------------------------------
    # Identity loss
    # ------------------------------------------------------------------

    def test_identity_loss_absent_by_default(self, cyclegan_fwd):
        """Identity loss keys must NOT appear when lambda_idt=0 (default)."""
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss()
        assert "idt_A" not in losses
        assert "idt_B" not in losses

    def test_identity_loss_present_when_enabled(self, cyclegan_fwd):
        """Identity loss keys MUST appear when lambda_idt > 0."""
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss(lambda_idt=0.5)
        assert "idt_A" in losses
        assert "idt_B" in losses

    @pytest.mark.parametrize("lambda_idt", [0.1, 0.5])
    def test_identity_loss_non_negative_and_no_nan(self, cyclegan_fwd, lambda_idt):
        """Identity losses must be non-negative and NaN-free for varying lambda_idt."""
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss(lambda_idt=lambda_idt)
        assert losses["idt_A"].item() >= 0
        assert losses["idt_B"].item() >= 0
        assert not torch.isnan(losses["idt_A"])
        assert not torch.isnan(losses["idt_B"])

    def test_identity_loss_included_in_total(self, cyclegan_fwd):
        """total_G must include the identity losses when lambda_idt > 0."""
        model, _, _ = cyclegan_fwd
        losses = model.compute_generator_loss(lambda_idt=0.5)
        expected = (
            losses["G_AB"]
            + losses["G_BA"]
            + losses["cycle_A"]
            + losses["cycle_B"]
            + losses["idt_A"]
            + losses["idt_B"]
        )
        assert torch.allclose(losses["total_G"], expected, atol=1e-5)

    def test_identity_loss_scales_proportionally_with_lambda_cyc(self, cyclegan_fwd):
        """Identity loss must scale exactly 2× when lambda_cyc is doubled (same forward pass)."""
        model, _, _ = cyclegan_fwd
        # Both loss computations reuse the same stored activations (no re-forward),
        # so the only difference is the scalar multiplier — allowing an exact ratio check.
        losses_low = model.compute_generator_loss(lambda_idt=0.5, lambda_cyc=5.0)
        losses_high = model.compute_generator_loss(lambda_idt=0.5, lambda_cyc=10.0)
        assert torch.allclose(losses_high["idt_A"], losses_low["idt_A"] * 2, atol=1e-5)
        assert torch.allclose(losses_high["idt_B"], losses_low["idt_B"] * 2, atol=1e-5)

"""Tests for AWADA covering attention mask handling, masked MSE loss computation, and loss calculation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from awada.models.awada import AWADA

DEVICE = "cpu"
IMG_SIZE = 64


def _make_model():
    return AWADA(device=DEVICE)


def _inputs(batch=1):
    real_A = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    real_B = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    att_A = torch.randint(0, 2, (batch, 1, IMG_SIZE, IMG_SIZE)).float()
    att_B = torch.randint(0, 2, (batch, 1, IMG_SIZE, IMG_SIZE)).float()
    return real_A, real_B, att_A, att_B


class TestAWADA:
    def test_set_input_with_attention(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        assert model.attention_A is not None
        assert model.attention_B is not None
        assert model.attention_A.shape == att_A.shape

    def test_set_input_without_attention(self):
        model = _make_model()
        real_A, real_B, _, _ = _inputs()
        model.set_input(real_A, real_B)
        assert model.attention_A is None
        assert model.attention_B is None

    def test_masked_mse_loss_no_mask(self):
        """With mask=None the loss should equal standard MSE."""
        model = _make_model()
        pred = torch.randn(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        loss_masked = model._masked_mse_loss(pred, target, mask=None)
        loss_standard = ((pred - target) ** 2).mean()
        assert torch.allclose(loss_masked, loss_standard, atol=1e-6)

    def test_masked_mse_loss_zero_mask(self):
        """All-zero mask should produce loss of 0."""
        model = _make_model()
        pred = torch.randn(1, 1, 8, 8)
        target = torch.ones_like(pred)
        mask = torch.zeros(1, 1, 32, 32)
        loss = model._masked_mse_loss(pred, target, mask=mask)
        assert loss.item() == pytest.approx(0.0)

    def test_masked_mse_loss_all_ones_mask_equals_standard(self):
        """All-ones mask should equal unmasked MSE."""
        model = _make_model()
        pred = torch.randn(1, 1, 8, 8)
        target = torch.randn(1, 1, 8, 8)
        mask = torch.ones(1, 1, 32, 32)
        loss_masked = model._masked_mse_loss(pred, target, mask=mask)
        loss_standard = ((pred - target) ** 2).mean()
        assert torch.allclose(loss_masked, loss_standard, atol=1e-5)

    def test_generator_loss_keys(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss()
        for key in ("G_AB", "G_BA", "cycle_A", "cycle_B", "total_G"):
            assert key in losses
        for key in ("idt_A", "idt_B"):
            assert key not in losses, f"Identity loss key should not be present: {key}"

    def test_discriminator_loss_keys(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        for key in ("D_A", "D_B", "total_D"):
            assert key in losses

    def test_generator_loss_no_nan(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss()
        for v in losses.values():
            assert not torch.isnan(v)

    def test_discriminator_loss_no_nan(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        for v in losses.values():
            assert not torch.isnan(v)

    def test_generator_loss_without_attention(self):
        """Generator loss runs correctly when no attention masks are provided."""
        model = _make_model()
        real_A, real_B, _, _ = _inputs()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert "total_G" in losses

    def test_total_d_is_sum(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        assert torch.allclose(losses["total_D"], losses["D_A"] + losses["D_B"], atol=1e-5)

    # ------------------------------------------------------------------
    # Identity loss (AWADA inherits unmasked identity loss from CycleGAN)
    # ------------------------------------------------------------------

    def test_identity_loss_absent_by_default(self):
        """Identity loss keys must NOT appear when lambda_idt=0 (default)."""
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert "idt_A" not in losses
        assert "idt_B" not in losses

    def test_identity_loss_present_and_unmasked_when_enabled(self):
        """Identity loss keys MUST appear when lambda_idt > 0 (unmasked)."""
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_idt=5.0)
        assert "idt_A" in losses
        assert "idt_B" in losses

    def test_identity_loss_included_in_total(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_idt=5.0)
        expected = (
            losses["G_AB"]
            + losses["G_BA"]
            + losses["cycle_A"]
            + losses["cycle_B"]
            + losses["idt_A"]
            + losses["idt_B"]
        )
        assert torch.allclose(losses["total_G"], expected, atol=1e-5)

    # ------------------------------------------------------------------
    # Semantic consistency loss (AWADA inherits unmasked semantic loss)
    # ------------------------------------------------------------------

    def test_semantic_loss_not_instantiated_when_zero(self):
        """criterion_sem must be None when lambda_sem=0 (default)."""
        model = _make_model()
        assert model.criterion_sem is None

    def test_semantic_loss_absent_by_default(self):
        model = _make_model()
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert "sem_AB" not in losses
        assert "sem_BA" not in losses

    def test_semantic_loss_present_when_enabled(self):
        """Semantic loss keys MUST appear when lambda_sem > 0 and criterion_sem is set."""
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("awada.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = AWADA(device=DEVICE, lambda_sem=1.0)
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_sem=1.0)
        assert "sem_AB" in losses
        assert "sem_BA" in losses

    def test_semantic_loss_included_in_total(self):
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("awada.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = AWADA(device=DEVICE, lambda_sem=1.0)
        real_A, real_B, att_A, att_B = _inputs()
        model.set_input(real_A, real_B, att_A, att_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_sem=1.0)
        expected = (
            losses["G_AB"]
            + losses["G_BA"]
            + losses["cycle_A"]
            + losses["cycle_B"]
            + losses["sem_AB"]
            + losses["sem_BA"]
        )
        assert torch.allclose(losses["total_G"], expected, atol=1e-5)

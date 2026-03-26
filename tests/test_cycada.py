"""Tests for CyCada covering semantic consistency loss and inheritance from CycleGAN."""

from unittest.mock import MagicMock, patch

import torch

from src.models.cycada import CyCada

DEVICE = "cpu"
IMG_SIZE = 64


def _make_cycada(lambda_sem=0.0):
    return CyCada(device=DEVICE, lambda_sem=lambda_sem)


def _real_pair(batch=1):
    real_A = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    real_B = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    return real_A, real_B


class TestCyCada:
    def test_inherits_cyclegan_forward(self):
        """CyCada forward pass should produce fake and reconstructed images."""
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        assert model.fake_B.shape == real_A.shape
        assert model.fake_A.shape == real_B.shape
        assert model.rec_A.shape == real_A.shape
        assert model.rec_B.shape == real_B.shape

    def test_inherits_discriminator_loss(self):
        """CyCada inherits discriminator loss from CycleGAN unchanged."""
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_discriminator_loss()
        for key in ("D_A", "D_B", "total_D"):
            assert key in losses

    # ------------------------------------------------------------------
    # Semantic consistency loss
    # ------------------------------------------------------------------

    def test_semantic_loss_not_instantiated_when_zero(self):
        """criterion_sem must be None when lambda_sem=0 (default)."""
        model = _make_cycada(lambda_sem=0.0)
        assert model.criterion_sem is None

    def test_semantic_loss_absent_by_default(self):
        """Semantic loss keys must NOT appear when lambda_sem=0 (default)."""
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert "sem_AB" not in losses
        assert "sem_BA" not in losses

    def test_semantic_loss_present_when_enabled(self):
        """Semantic loss keys MUST appear when lambda_sem > 0 and criterion_sem is set."""
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("src.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = CyCada(device=DEVICE, lambda_sem=1.0)
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_sem=1.0)
        assert "sem_AB" in losses
        assert "sem_BA" in losses

    def test_semantic_loss_instantiated_when_nonzero(self):
        """criterion_sem must NOT be None when lambda_sem > 0."""
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("src.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = CyCada(device=DEVICE, lambda_sem=1.0)
        assert model.criterion_sem is not None

    def test_semantic_loss_included_in_total(self):
        """total_G must include the semantic losses when lambda_sem > 0."""
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("src.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = CyCada(device=DEVICE, lambda_sem=1.0)
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
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

    def test_semantic_loss_non_negative(self):
        mock_sem = MagicMock(return_value=torch.tensor(0.5))
        with patch("src.models.cycada.SemanticConsistencyLoss", return_value=mock_sem):
            model = CyCada(device=DEVICE, lambda_sem=1.0)
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_sem=1.0)
        assert losses["sem_AB"].item() >= 0
        assert losses["sem_BA"].item() >= 0

    def test_no_semantic_loss_without_criterion(self):
        """When criterion_sem is None, semantic keys must not appear even if lambda_sem > 0."""
        """When criterion_sem is None, semantic keys must not appear even if lambda_sem > 0."""
        model = _make_cycada(lambda_sem=0.0)
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_sem=1.0)
        assert "sem_AB" not in losses
        assert "sem_BA" not in losses

    def test_generator_loss_keys_without_sem(self):
        """Standard loss keys must be present even without semantic loss."""
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        for key in ("G_AB", "G_BA", "cycle_A", "cycle_B", "total_G"):
            assert key in losses

    def test_no_nan_in_losses(self):
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        g_losses = model.compute_generator_loss()
        d_losses = model.compute_discriminator_loss()
        for v in list(g_losses.values()) + list(d_losses.values()):
            assert not torch.isnan(v), "NaN detected in loss"

    # ------------------------------------------------------------------
    # Identity loss inherited from CycleGAN
    # ------------------------------------------------------------------

    def test_identity_loss_absent_by_default(self):
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss()
        assert "idt_A" not in losses
        assert "idt_B" not in losses

    def test_identity_loss_present_when_enabled(self):
        model = _make_cycada()
        real_A, real_B = _real_pair()
        model.set_input(real_A, real_B)
        model.forward()
        losses = model.compute_generator_loss(lambda_idt=5.0)
        assert "idt_A" in losses
        assert "idt_B" in losses

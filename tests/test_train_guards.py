"""Tests for NaN/Inf safety guards in training loops."""

from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_param_list():
    """Return a list with a single dummy parameter (no gradients needed)."""
    p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    return [p]


def _make_mock_model():
    """Build a minimal MagicMock that mimics the CycleGAN-like model interface."""
    model = MagicMock()
    model.G_AB.parameters.return_value = iter(_make_param_list())
    model.G_BA.parameters.return_value = iter(_make_param_list())
    model.D_A.parameters.return_value = iter(_make_param_list())
    model.D_B.parameters.return_value = iter(_make_param_list())
    return model


def _run_train_iteration(g_losses, d_losses, script_module):
    """
    Simulate one training iteration using the guard logic extracted from the
    training scripts.  ``script_module`` is one of the three train_* modules.

    We re-implement only the guarded section so we can exercise the exact same
    ``torch.isfinite`` checks that live in the training loops.
    """
    epoch = 0
    iteration = 0

    # Generator guard (mirrors code in all three training scripts)
    for loss_name, loss_val in g_losses.items():
        if not torch.isfinite(loss_val):
            raise RuntimeError(
                f"Non-finite generator loss '{loss_name}' detected at epoch {epoch + 1}, "
                f"iteration {iteration + 1}: {loss_val.item()}"
            )

    # Discriminator guard
    for loss_name, loss_val in d_losses.items():
        if not torch.isfinite(loss_val):
            raise RuntimeError(
                f"Non-finite discriminator loss '{loss_name}' detected at epoch {epoch + 1}, "
                f"iteration {iteration + 1}: {loss_val.item()}"
            )


class TestNaNGuardLogic:
    """Unit tests for the NaN/Inf guard logic pattern used in all three training scripts."""

    def test_finite_losses_do_not_raise(self):
        g = {"total_G": torch.tensor(0.5), "G_AB": torch.tensor(0.3), "cycle_A": torch.tensor(0.2)}
        d = {"total_D": torch.tensor(0.4), "D_A": torch.tensor(0.2)}
        _run_train_iteration(g, d, None)  # must not raise

    def test_nan_generator_total_raises_runtime_error(self):
        g = {"total_G": torch.tensor(float("nan")), "G_AB": torch.tensor(0.3)}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="generator loss 'total_G'"):
            _run_train_iteration(g, d, None)

    def test_inf_generator_loss_raises_runtime_error(self):
        g = {"total_G": torch.tensor(float("inf")), "G_AB": torch.tensor(0.3)}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="generator loss 'total_G'"):
            _run_train_iteration(g, d, None)

    def test_neg_inf_generator_loss_raises_runtime_error(self):
        g = {"G_AB": torch.tensor(float("-inf")), "total_G": torch.tensor(float("-inf"))}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="generator loss"):
            _run_train_iteration(g, d, None)

    def test_nan_discriminator_total_raises_runtime_error(self):
        g = {"total_G": torch.tensor(0.5), "G_AB": torch.tensor(0.3)}
        d = {"total_D": torch.tensor(float("nan")), "D_A": torch.tensor(0.2)}
        with pytest.raises(RuntimeError, match="discriminator loss 'total_D'"):
            _run_train_iteration(g, d, None)

    def test_inf_discriminator_loss_raises_runtime_error(self):
        g = {"total_G": torch.tensor(0.5)}
        d = {"D_A": torch.tensor(float("inf")), "total_D": torch.tensor(float("inf"))}
        with pytest.raises(RuntimeError, match="discriminator loss"):
            _run_train_iteration(g, d, None)

    def test_error_message_contains_epoch_and_iteration(self):
        g = {"total_G": torch.tensor(float("nan"))}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match=r"epoch 1.*iteration 1"):
            _run_train_iteration(g, d, None)

    def test_error_message_contains_loss_value(self):
        g = {"total_G": torch.tensor(float("nan"))}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="nan"):
            _run_train_iteration(g, d, None)

    def test_nan_in_sub_loss_raises(self):
        """A NaN in a sub-loss (not just total_G) must also trigger the guard."""
        g = {"G_AB": torch.tensor(float("nan")), "total_G": torch.tensor(float("nan"))}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="generator loss"):
            _run_train_iteration(g, d, None)

    def test_zero_losses_do_not_raise(self):
        g = {"total_G": torch.tensor(0.0), "cycle_A": torch.tensor(0.0)}
        d = {"total_D": torch.tensor(0.0)}
        _run_train_iteration(g, d, None)  # must not raise


class TestTrainScriptGuardsIntegration:
    """
    Integration-style tests that patch just enough of the training scripts to
    drive the NaN/Inf guards without touching real datasets or models.
    """

    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    # ------------------------------------------------------------------
    # train_cyclegan
    # ------------------------------------------------------------------

    def test_train_cyclegan_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")),
            "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch(
                "tools.train_cyclegan.CycleGAN",
                return_value=mock_model,
            ),
            patch(
                "tools.train_cyclegan.DataLoader",
                return_value=self._make_dataloader_mock(real_A, real_B),
            ),
            patch(
                "tools.train_cyclegan.UnpairedImageDataset",
            ),
            patch(
                "sys.argv",
                [
                    "train_cyclegan.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                cyclegan_main()

    def test_train_cyclegan_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1),
            "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch(
                "tools.train_cyclegan.DataLoader",
                return_value=self._make_dataloader_mock(real_A, real_B),
            ),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
            patch(
                "sys.argv",
                [
                    "train_cyclegan.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                cyclegan_main()

    # ------------------------------------------------------------------
    # train_cycada
    # ------------------------------------------------------------------

    def test_train_cycada_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cycada import main as cycada_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")),
            "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch(
                "tools.train_cycada.DataLoader",
                return_value=self._make_dataloader_mock(real_A, real_B),
            ),
            patch("tools.train_cycada.UnpairedImageDataset"),
            patch(
                "sys.argv",
                [
                    "train_cycada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                cycada_main()

    def test_train_cycada_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cycada import main as cycada_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1),
            "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch(
                "tools.train_cycada.DataLoader",
                return_value=self._make_dataloader_mock(real_A, real_B),
            ),
            patch("tools.train_cycada.UnpairedImageDataset"),
            patch(
                "sys.argv",
                [
                    "train_cycada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                cycada_main()

    # ------------------------------------------------------------------
    # train_awada
    # ------------------------------------------------------------------

    def test_train_awada_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)

        from tools.train_awada import main as awada_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")),
            "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch(
                "tools.train_awada.DataLoader",
                return_value=self._make_dataloader_mock(
                    real_A, real_B, att_A, att_B, with_attention=True
                ),
            ),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                awada_main()

    def test_train_awada_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)

        from tools.train_awada import main as awada_main

        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1),
            "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")),
            "D_A": torch.tensor(0.2),
            "D_B": torch.tensor(0.2),
        }

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch(
                "tools.train_awada.DataLoader",
                return_value=self._make_dataloader_mock(
                    real_A, real_B, att_A, att_B, with_attention=True
                ),
            ),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                awada_main()

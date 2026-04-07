"""Tests for NaN/Inf safety guards and resumable checkpointing in training loops."""

import os
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
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
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
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
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

    def test_train_awada_raises_when_target_attention_dir_missing(self, tmp_path):
        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            with pytest.raises(SystemExit):
                awada_main()


def _make_minimal_checkpoint(tmp_path, prefix="cyclegan"):
    """Create a minimal checkpoint file with all required keys and return its path."""
    # Use 2 parameters per optimizer to match the training scripts, which build
    # opt_G from list(G_AB.parameters()) + list(G_BA.parameters()) and
    # opt_D from list(D_A.parameters()) + list(D_B.parameters()).
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p1, p2], lr=1e-4, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: 1.0)

    ckpt = {
        "epoch": 3,
        "G_AB": {},
        "G_BA": {},
        "D_A": {},
        "D_B": {},
        "opt_G": opt.state_dict(),
        "opt_D": opt.state_dict(),
        "sched_G": sched.state_dict(),
        "sched_D": sched.state_dict(),
    }
    path = os.path.join(str(tmp_path), f"{prefix}_epoch_3.pth")
    torch.save(ckpt, path)
    return path


class TestResumeCheckpointing:
    """Tests that the --resume flag correctly loads checkpoint state in training scripts."""

    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    def test_cyclegan_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        """Saved checkpoint must include opt_G, opt_D, sched_G, sched_D keys."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        mock_model = MagicMock()
        mock_model.G_AB.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        mock_model.G_BA.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        mock_model.D_A.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        mock_model.D_B.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        mock_model.G_AB.state_dict.return_value = {}
        mock_model.G_BA.state_dict.return_value = {}
        mock_model.D_A.state_dict.return_value = {}
        mock_model.D_B.state_dict.return_value = {}
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4, requires_grad=True),
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
            cyclegan_main()

        ckpt_path = os.path.join(str(tmp_path), "cyclegan_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"

    def test_cyclegan_resume_loads_start_epoch(self, tmp_path):
        """--resume should set start_epoch from checkpoint so the loop starts at epoch 3."""
        ckpt_path = _make_minimal_checkpoint(tmp_path, "cyclegan")
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        mock_model = MagicMock()
        for attr in ("G_AB", "G_BA", "D_A", "D_B"):
            getattr(mock_model, attr).parameters.return_value = iter(
                [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
            )
            getattr(mock_model, attr).state_dict.return_value = {}
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4, requires_grad=True),
        }

        epochs_seen = []

        original_tqdm = __import__("tqdm").tqdm

        def tracking_tqdm(iterable, desc=""):
            epochs_seen.append(desc)
            return original_tqdm(iterable, desc=desc)

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch(
                "tools.train_cyclegan.DataLoader",
                return_value=self._make_dataloader_mock(real_A, real_B),
            ),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
            patch("tools.train_cyclegan.tqdm", side_effect=tracking_tqdm),
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
                    "5",
                    "--resume",
                    ckpt_path,
                ],
            ),
        ):
            cyclegan_main()

        # With start_epoch=3 and total epochs=5, we expect epochs 4 and 5 (indices 3 and 4)
        assert len(epochs_seen) == 2
        assert "4/5" in epochs_seen[0]
        assert "5/5" in epochs_seen[1]

    def test_cycada_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        """Saved CyCada checkpoint must include opt_G, opt_D, sched_G, sched_D keys."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cycada import main as cycada_main

        mock_model = MagicMock()
        for attr in ("G_AB", "G_BA", "D_A", "D_B"):
            getattr(mock_model, attr).parameters.return_value = iter(
                [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
            )
            getattr(mock_model, attr).state_dict.return_value = {}
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4, requires_grad=True),
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
            cycada_main()

        ckpt_path = os.path.join(str(tmp_path), "cycada_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"

    def test_awada_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        """Saved AWADA checkpoint must include opt_G, opt_D, sched_G, sched_D keys."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)

        from tools.train_awada import main as awada_main

        mock_model = MagicMock()
        for attr in ("G_AB", "G_BA", "D_A", "D_B"):
            getattr(mock_model, attr).parameters.return_value = iter(
                [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
            )
            getattr(mock_model, attr).state_dict.return_value = {}
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True),
            "cycle_A": torch.tensor(0.2),
            "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4, requires_grad=True),
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
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                ],
            ),
        ):
            awada_main()

        ckpt_path = os.path.join(str(tmp_path), "awada_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"


def _make_full_mock_model():
    """Build a MagicMock model with all state_dict methods configured."""
    mock_model = MagicMock()
    for attr in ("G_AB", "G_BA", "D_A", "D_B"):
        getattr(mock_model, attr).parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        getattr(mock_model, attr).state_dict.return_value = {}
    mock_model.compute_generator_loss.return_value = {
        "total_G": torch.tensor(0.5, requires_grad=True),
        "cycle_A": torch.tensor(0.2),
        "cycle_B": torch.tensor(0.2),
    }
    mock_model.compute_discriminator_loss.return_value = {
        "total_D": torch.tensor(0.4, requires_grad=True),
    }
    return mock_model


class TestSaveEvery:
    """Tests for the --save_every checkpoint frequency argument."""

    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    # ------------------------------------------------------------------
    # train_cyclegan
    # ------------------------------------------------------------------

    def test_cyclegan_save_every_saves_at_multiples(self, tmp_path):
        """With --save_every 2 and --epochs 5, checkpoints should be saved at epochs 2, 4, 5."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
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
                    "5",
                    "--save_every",
                    "2",
                ],
            ),
        ):
            cyclegan_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cyclegan_epoch_")}
        assert saved == {"cyclegan_epoch_2.pth", "cyclegan_epoch_4.pth", "cyclegan_epoch_5.pth"}

    def test_cyclegan_default_save_every_saves_only_final_epoch(self, tmp_path):
        """Default --save_every 10 with --epochs 3 should only save the final epoch."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
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
                    "3",
                ],
            ),
        ):
            cyclegan_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cyclegan_epoch_")}
        assert saved == {"cyclegan_epoch_3.pth"}

    # ------------------------------------------------------------------
    # train_cycada
    # ------------------------------------------------------------------

    def test_cycada_save_every_saves_at_multiples(self, tmp_path):
        """With --save_every 2 and --epochs 5, checkpoints should be saved at epochs 2, 4, 5."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=_make_full_mock_model()),
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
                    "5",
                    "--save_every",
                    "2",
                ],
            ),
        ):
            cycada_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cycada_epoch_")}
        assert saved == {"cycada_epoch_2.pth", "cycada_epoch_4.pth", "cycada_epoch_5.pth"}

    def test_cycada_default_save_every_saves_only_final_epoch(self, tmp_path):
        """Default --save_every 10 with --epochs 3 should only save the final epoch."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=_make_full_mock_model()),
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
                    "3",
                ],
            ),
        ):
            cycada_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cycada_epoch_")}
        assert saved == {"cycada_epoch_3.pth"}

    # ------------------------------------------------------------------
    # train_awada
    # ------------------------------------------------------------------

    def test_awada_save_every_saves_at_multiples(self, tmp_path):
        """With --save_every 2 and --epochs 5, checkpoints should be saved at epochs 2, 4, 5."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=_make_full_mock_model()),
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
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "5",
                    "--save_every",
                    "2",
                ],
            ),
        ):
            awada_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("awada_epoch_")}
        assert saved == {"awada_epoch_2.pth", "awada_epoch_4.pth", "awada_epoch_5.pth"}

    def test_awada_default_save_every_saves_only_final_epoch(self, tmp_path):
        """Default --save_every 10 with --epochs 3 should only save the final epoch."""
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=_make_full_mock_model()),
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
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "3",
                ],
            ),
        ):
            awada_main()

        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("awada_epoch_")}
        assert saved == {"awada_epoch_3.pth"}


def _make_full_mock_model_with_side_effects(g_losses_seq, d_losses_seq):
    """Build a mock model with per-iteration varying G and D losses."""
    mock_model = MagicMock()
    for attr in ("G_AB", "G_BA", "D_A", "D_B"):
        getattr(mock_model, attr).parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        getattr(mock_model, attr).state_dict.return_value = {}
    mock_model.compute_generator_loss.side_effect = g_losses_seq
    mock_model.compute_discriminator_loss.side_effect = d_losses_seq
    return mock_model


class TestRunningLossLogging:
    """Tests that per-interval logs use running averages over log_interval iterations."""

    def _cyclegan_batches(self, n):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        return [(real_A, real_B)] * n

    def _awada_batches(self, n):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        return [(real_A, real_B, att_A, att_B)] * n

    def _make_g_d_losses(self, g_vals, d_vals):
        g_seq = [
            {
                "total_G": torch.tensor(v, requires_grad=True),
                "cycle_A": torch.tensor(0.1),
                "cycle_B": torch.tensor(0.1),
            }
            for v in g_vals
        ]
        d_seq = [{"total_D": torch.tensor(v, requires_grad=True)} for v in d_vals]
        return g_seq, d_seq

    # ------------------------------------------------------------------
    # train_cyclegan
    # ------------------------------------------------------------------

    def test_cyclegan_interval_log_uses_running_average(self, tmp_path, caplog):
        """[Epoch X, Iter Y] log must show the running average G/D loss, not the instantaneous value."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        # Two batches with different G losses; average = (1.0 + 3.0) / 2 = 2.0
        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 1.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cyclegan_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        # Running average: G = (1.0 + 3.0) / 2 = 2.000, D = (0.5 + 1.5) / 2 = 1.000
        assert "G=2.000" in interval_logs[0]
        assert "D=1.000" in interval_logs[0]

    def test_cyclegan_running_loss_resets_after_interval(self, tmp_path, caplog):
        """Running accumulators must reset after each log interval."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        # Four batches: interval 1 avg=(1+3)/2=2.0, interval 2 avg=(5+7)/2=6.0
        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0, 5.0, 7.0], [0.1] * 4)
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._cyclegan_batches(4)),
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
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cyclegan_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 2
        assert "G=2.000" in interval_logs[0]
        assert "G=6.000" in interval_logs[1]

    # ------------------------------------------------------------------
    # train_cycada
    # ------------------------------------------------------------------

    def test_cycada_interval_log_uses_running_average(self, tmp_path, caplog):
        """[Epoch X, Iter Y] log must show the running average G/D loss, not the instantaneous value."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 1.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cycada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "G=2.000" in interval_logs[0]
        assert "D=1.000" in interval_logs[0]

    def test_cycada_running_loss_resets_after_interval(self, tmp_path, caplog):
        """Running accumulators must reset after each log interval."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0, 5.0, 7.0], [0.1] * 4)
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._cyclegan_batches(4)),
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
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cycada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 2
        assert "G=2.000" in interval_logs[0]
        assert "G=6.000" in interval_logs[1]

    # ------------------------------------------------------------------
    # train_awada
    # ------------------------------------------------------------------

    def test_awada_interval_log_uses_running_average(self, tmp_path, caplog):
        """[Epoch X, Iter Y] log must show the running average G/D loss, not the instantaneous value."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 1.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._awada_batches(2)),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            awada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "G=2.000" in interval_logs[0]
        assert "D=1.000" in interval_logs[0]

    def test_awada_running_loss_resets_after_interval(self, tmp_path, caplog):
        """Running accumulators must reset after each log interval."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0, 5.0, 7.0], [0.1] * 4)
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._awada_batches(4)),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            awada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 2
        assert "G=2.000" in interval_logs[0]
        assert "G=6.000" in interval_logs[1]

    # ------------------------------------------------------------------
    # Semantic consistency loss logging (train_cycada and train_awada)
    # ------------------------------------------------------------------

    def _make_g_d_losses_with_sem(self, g_vals, sem_vals, d_vals):
        """Build mock loss sequences that include sem_AB / sem_BA keys."""
        g_seq = [
            {
                "total_G": torch.tensor(g, requires_grad=True),
                "cycle_A": torch.tensor(0.1),
                "cycle_B": torch.tensor(0.1),
                "sem_AB": torch.tensor(s / 2),
                "sem_BA": torch.tensor(s / 2),
            }
            for g, s in zip(g_vals, sem_vals)
        ]
        d_seq = [{"total_D": torch.tensor(v, requires_grad=True)} for v in d_vals]
        return g_seq, d_seq

    def test_cycada_sem_loss_logged_when_lambda_sem_nonzero(self, tmp_path, caplog):
        """When lambda_sem > 0, interval log must include sem= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        # sem_AB=0.2, sem_BA=0.2 per iter -> combined sem = 0.4; avg over 2 = 0.4
        g_seq, d_seq = self._make_g_d_losses_with_sem([1.0, 3.0], [0.4, 0.4], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--lambda_sem",
                    "1.0",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cycada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "sem=" in interval_logs[0]
        assert "sem=0.400" in interval_logs[0]

    def test_cycada_sem_loss_not_logged_when_lambda_sem_zero(self, tmp_path, caplog):
        """When lambda_sem = 0, interval log must NOT include sem= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--lambda_sem",
                    "0.0",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cycada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "sem=" not in interval_logs[0]

    def test_cycada_epoch_summary_includes_sem_when_enabled(self, tmp_path, caplog):
        """Epoch-end summary must include avg sem loss= when lambda_sem > 0."""
        import logging

        g_seq, d_seq = self._make_g_d_losses_with_sem([1.0], [0.6], [0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cycada import main as cycada_main

        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._cyclegan_batches(1)),
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
                    "--lambda_sem",
                    "1.0",
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cycada_main()

        epoch_logs = [
            r.message for r in caplog.records if "Epoch" in r.message and "complete" in r.message
        ]
        assert len(epoch_logs) == 1
        assert "avg sem loss=" in epoch_logs[0]

    def test_awada_sem_loss_logged_when_lambda_sem_nonzero(self, tmp_path, caplog):
        """When lambda_sem > 0, AWADA interval log must include sem= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses_with_sem([1.0, 3.0], [0.4, 0.4], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._awada_batches(2)),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                    "--lambda_sem",
                    "1.0",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            awada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "sem=" in interval_logs[0]
        assert "sem=0.400" in interval_logs[0]

    def test_awada_sem_loss_not_logged_when_lambda_sem_zero(self, tmp_path, caplog):
        """When lambda_sem = 0, AWADA interval log must NOT include sem= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._awada_batches(2)),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                    "--lambda_sem",
                    "0.0",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            awada_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "sem=" not in interval_logs[0]

    def test_awada_epoch_summary_includes_sem_when_enabled(self, tmp_path, caplog):
        """Epoch-end summary must include avg sem loss= when lambda_sem > 0."""
        import logging

        g_seq, d_seq = self._make_g_d_losses_with_sem([1.0], [0.6], [0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_awada import main as awada_main

        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._awada_batches(1)),
            patch("tools.train_awada.AttentionPairedDataset"),
            patch(
                "sys.argv",
                [
                    "train_awada.py",
                    "--source_dir",
                    str(tmp_path),
                    "--target_dir",
                    str(tmp_path),
                    "--source_attention_dir",
                    str(tmp_path),
                    "--target_attention_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(tmp_path),
                    "--epochs",
                    "1",
                    "--lambda_sem",
                    "1.0",
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            awada_main()

        epoch_logs = [
            r.message for r in caplog.records if "Epoch" in r.message and "complete" in r.message
        ]
        assert len(epoch_logs) == 1
        assert "avg sem loss=" in epoch_logs[0]

    # ------------------------------------------------------------------
    # Identity loss logging (train_cyclegan)
    # ------------------------------------------------------------------

    def _make_g_d_losses_with_idt(self, g_vals, idt_vals, d_vals):
        """Build mock loss sequences that include idt_A / idt_B keys."""
        g_seq = [
            {
                "total_G": torch.tensor(g, requires_grad=True),
                "cycle_A": torch.tensor(0.1),
                "cycle_B": torch.tensor(0.1),
                "idt_A": torch.tensor(i / 2),
                "idt_B": torch.tensor(i / 2),
            }
            for g, i in zip(g_vals, idt_vals)
        ]
        d_seq = [{"total_D": torch.tensor(v, requires_grad=True)} for v in d_vals]
        return g_seq, d_seq

    def test_cyclegan_idt_loss_logged_when_lambda_idt_nonzero(self, tmp_path, caplog):
        """When lambda_idt > 0, CycleGAN interval log must include idt= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        # idt_A=0.2, idt_B=0.2 per iter -> combined idt = 0.4; avg over 2 = 0.4
        g_seq, d_seq = self._make_g_d_losses_with_idt([1.0, 3.0], [0.4, 0.4], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--lambda_idt",
                    "0.5",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cyclegan_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "idt=" in interval_logs[0]
        assert "idt=0.400" in interval_logs[0]

    def test_cyclegan_idt_loss_not_logged_when_lambda_idt_zero(self, tmp_path, caplog):
        """When lambda_idt = 0, CycleGAN interval log must NOT include idt= field."""
        import logging

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("log_interval: 2\n")

        g_seq, d_seq = self._make_g_d_losses([1.0, 3.0], [0.5, 0.5])
        mock_model = _make_full_mock_model_with_side_effects(g_seq, d_seq)

        from tools.train_cyclegan import main as cyclegan_main

        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._cyclegan_batches(2)),
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
                    "--lambda_idt",
                    "0.0",
                    "--config",
                    str(config_path),
                ],
            ),
            caplog.at_level(logging.INFO),
        ):
            cyclegan_main()

        interval_logs = [
            r.message for r in caplog.records if "[Epoch" in r.message and "Iter" in r.message
        ]
        assert len(interval_logs) == 1
        assert "idt=" not in interval_logs[0]

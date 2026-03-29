"""Tests for NaN/Inf safety guards and resumable checkpointing in training loops."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf


def _make_param_list():
    p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    return [p]


def _make_mock_model():
    model = MagicMock()
    model.G_AB.parameters.return_value = iter(_make_param_list())
    model.G_BA.parameters.return_value = iter(_make_param_list())
    model.D_A.parameters.return_value = iter(_make_param_list())
    model.D_B.parameters.return_value = iter(_make_param_list())
    return model


def _make_gan_cfg(tmp_path, epochs=1, extra=None):
    cfg = OmegaConf.create({
        "training": {
            "epochs": epochs, "lr": 2e-4, "betas": [0.5, 0.999], "batch_size": 1,
            "patch_size": 64, "resize_min_side": 600, "seed": 42, "amp": False,
            "save_every": 10, "resume": None, "log_interval": 100,
        },
        "model": {
            "lambda_gan": 1.0, "lambda_cyc": 10.0, "lambda_idt": 0.0, "lambda_sem": 0.0,
            "buffer_size": 50, "buffer_return_prob": 0.5, "disc_loss_avg_factor": 0.5,
        },
        "data": {"source_dir": str(tmp_path), "target_dir": str(tmp_path), "output_dir": str(tmp_path)},
        "hardware": {"device": "cpu"},
    })
    if extra:
        cfg = OmegaConf.merge(cfg, extra)
    return cfg


def _make_awada_cfg(tmp_path, epochs=1, extra=None):
    cfg = _make_gan_cfg(tmp_path, epochs=epochs)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {
        "source_attention_dir": str(tmp_path), "target_attention_dir": str(tmp_path),
    }}))
    if extra:
        cfg = OmegaConf.merge(cfg, extra)
    return cfg


def _run_train_iteration(g_losses, d_losses, script_module):
    epoch = 0
    iteration = 0
    for loss_name, loss_val in g_losses.items():
        if not torch.isfinite(loss_val):
            raise RuntimeError(
                f"Non-finite generator loss '{loss_name}' detected at epoch {epoch + 1}, "
                f"iteration {iteration + 1}: {loss_val.item()}"
            )
    for loss_name, loss_val in d_losses.items():
        if not torch.isfinite(loss_val):
            raise RuntimeError(
                f"Non-finite discriminator loss '{loss_name}' detected at epoch {epoch + 1}, "
                f"iteration {iteration + 1}: {loss_val.item()}"
            )


class TestNaNGuardLogic:
    def test_finite_losses_do_not_raise(self):
        g = {"total_G": torch.tensor(0.5), "G_AB": torch.tensor(0.3), "cycle_A": torch.tensor(0.2)}
        d = {"total_D": torch.tensor(0.4), "D_A": torch.tensor(0.2)}
        _run_train_iteration(g, d, None)

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
        g = {"G_AB": torch.tensor(float("nan")), "total_G": torch.tensor(float("nan"))}
        d = {"total_D": torch.tensor(0.4)}
        with pytest.raises(RuntimeError, match="generator loss"):
            _run_train_iteration(g, d, None)

    def test_zero_losses_do_not_raise(self):
        g = {"total_G": torch.tensor(0.0), "cycle_A": torch.tensor(0.0)}
        d = {"total_D": torch.tensor(0.0)}
        _run_train_iteration(g, d, None)


class TestTrainScriptGuardsIntegration:
    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    def test_train_cyclegan_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")), "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2), "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                cyclegan_train(cfg)

    def test_train_cyclegan_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True), "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1), "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=mock_model),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                cyclegan_train(cfg)

    def test_train_cycada_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cycada import _train as cycada_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")), "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2), "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cycada.UnpairedImageDataset"),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                cycada_train(cfg)

    def test_train_cycada_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cycada import _train as cycada_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True), "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1), "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cycada.CyCada", return_value=mock_model),
            patch("tools.train_cycada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cycada.UnpairedImageDataset"),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                cycada_train(cfg)

    def test_train_awada_raises_on_nan_generator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        from tools.train_awada import _train as awada_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(float("nan")), "G_AB": torch.tensor(0.5),
            "cycle_A": torch.tensor(0.2), "cycle_B": torch.tensor(0.2),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(0.4), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_awada_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B, att_A, att_B, with_attention=True)),
            patch("tools.train_awada.AttentionPairedDataset"),
        ):
            with pytest.raises(RuntimeError, match="generator loss"):
                awada_train(cfg)

    def test_train_awada_raises_on_nan_discriminator_loss(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        from tools.train_awada import _train as awada_train
        mock_model = _make_mock_model()
        mock_model.compute_generator_loss.return_value = {
            "total_G": torch.tensor(0.5, requires_grad=True), "G_AB": torch.tensor(0.3),
            "cycle_A": torch.tensor(0.1), "cycle_B": torch.tensor(0.1),
        }
        mock_model.compute_discriminator_loss.return_value = {
            "total_D": torch.tensor(float("nan")), "D_A": torch.tensor(0.2), "D_B": torch.tensor(0.2),
        }
        cfg = _make_awada_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_awada.AWADA", return_value=mock_model),
            patch("tools.train_awada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B, att_A, att_B, with_attention=True)),
            patch("tools.train_awada.AttentionPairedDataset"),
        ):
            with pytest.raises(RuntimeError, match="discriminator loss"):
                awada_train(cfg)

    def test_train_awada_requires_attention_dirs(self):
        from awada.conf import AwadaDataConfig
        from omegaconf import MISSING
        assert AwadaDataConfig.source_attention_dir == MISSING
        assert AwadaDataConfig.target_attention_dir == MISSING


def _make_minimal_checkpoint(tmp_path, prefix="cyclegan"):
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p1, p2], lr=1e-4, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: 1.0)
    ckpt = {
        "epoch": 3, "G_AB": {}, "G_BA": {}, "D_A": {}, "D_B": {},
        "opt_G": opt.state_dict(), "opt_D": opt.state_dict(),
        "sched_G": sched.state_dict(), "sched_D": sched.state_dict(),
    }
    path = os.path.join(str(tmp_path), f"{prefix}_epoch_3.pth")
    torch.save(ckpt, path)
    return path


def _make_full_mock_model():
    mock_model = MagicMock()
    for attr in ("G_AB", "G_BA", "D_A", "D_B"):
        getattr(mock_model, attr).parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
        )
        getattr(mock_model, attr).state_dict.return_value = {}
    mock_model.compute_generator_loss.return_value = {
        "total_G": torch.tensor(0.5, requires_grad=True),
        "cycle_A": torch.tensor(0.2), "cycle_B": torch.tensor(0.2),
    }
    mock_model.compute_discriminator_loss.return_value = {
        "total_D": torch.tensor(0.4, requires_grad=True),
    }
    return mock_model


class TestResumeCheckpointing:
    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    def test_cyclegan_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
        ):
            cyclegan_train(cfg)
        ckpt_path = os.path.join(str(tmp_path), "cyclegan_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"

    def test_cyclegan_resume_loads_start_epoch(self, tmp_path):
        ckpt_path = _make_minimal_checkpoint(tmp_path, "cyclegan")
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        epochs_seen = []
        original_tqdm = __import__("tqdm").tqdm

        def tracking_tqdm(iterable, desc=""):
            epochs_seen.append(desc)
            return original_tqdm(iterable, desc=desc)

        cfg = _make_gan_cfg(tmp_path, epochs=5)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"resume": ckpt_path}}))
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
            patch("tools.train_cyclegan.tqdm", side_effect=tracking_tqdm),
        ):
            cyclegan_train(cfg)
        assert len(epochs_seen) == 2
        assert "4/5" in epochs_seen[0]
        assert "5/5" in epochs_seen[1]

    def test_cycada_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cycada import _train as cycada_train
        cfg = _make_gan_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_cycada.CyCada", return_value=_make_full_mock_model()),
            patch("tools.train_cycada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cycada.UnpairedImageDataset"),
        ):
            cycada_train(cfg)
        ckpt_path = os.path.join(str(tmp_path), "cycada_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt

    def test_awada_checkpoint_contains_optimizer_and_scheduler_states(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        from tools.train_awada import _train as awada_train
        cfg = _make_awada_cfg(tmp_path, epochs=1)
        with (
            patch("tools.train_awada.AWADA", return_value=_make_full_mock_model()),
            patch("tools.train_awada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B, att_A, att_B, with_attention=True)),
            patch("tools.train_awada.AttentionPairedDataset"),
        ):
            awada_train(cfg)
        ckpt_path = os.path.join(str(tmp_path), "awada_epoch_1.pth")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "G_AB", "G_BA", "D_A", "D_B", "opt_G", "opt_D", "sched_G", "sched_D"):
            assert key in ckpt


class TestSaveEvery:
    def _make_dataloader_mock(self, real_A, real_B, att_A=None, att_B=None, with_attention=False):
        batch = (real_A, real_B, att_A, att_B) if with_attention else (real_A, real_B)
        return [batch]

    def test_cyclegan_save_every_saves_at_multiples(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        cfg = _make_gan_cfg(tmp_path, epochs=5)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"save_every": 2}}))
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
        ):
            cyclegan_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cyclegan_epoch_")}
        assert saved == {"cyclegan_epoch_2.pth", "cyclegan_epoch_4.pth", "cyclegan_epoch_5.pth"}

    def test_cyclegan_default_save_every_saves_only_final_epoch(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cyclegan import _train as cyclegan_train
        cfg = _make_gan_cfg(tmp_path, epochs=3)
        with (
            patch("tools.train_cyclegan.CycleGAN", return_value=_make_full_mock_model()),
            patch("tools.train_cyclegan.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cyclegan.UnpairedImageDataset"),
        ):
            cyclegan_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cyclegan_epoch_")}
        assert saved == {"cyclegan_epoch_3.pth"}

    def test_cycada_save_every_saves_at_multiples(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cycada import _train as cycada_train
        cfg = _make_gan_cfg(tmp_path, epochs=5)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"save_every": 2}}))
        with (
            patch("tools.train_cycada.CyCada", return_value=_make_full_mock_model()),
            patch("tools.train_cycada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cycada.UnpairedImageDataset"),
        ):
            cycada_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cycada_epoch_")}
        assert saved == {"cycada_epoch_2.pth", "cycada_epoch_4.pth", "cycada_epoch_5.pth"}

    def test_cycada_default_save_every_saves_only_final_epoch(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        from tools.train_cycada import _train as cycada_train
        cfg = _make_gan_cfg(tmp_path, epochs=3)
        with (
            patch("tools.train_cycada.CyCada", return_value=_make_full_mock_model()),
            patch("tools.train_cycada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B)),
            patch("tools.train_cycada.UnpairedImageDataset"),
        ):
            cycada_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("cycada_epoch_")}
        assert saved == {"cycada_epoch_3.pth"}

    def test_awada_save_every_saves_at_multiples(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        from tools.train_awada import _train as awada_train
        cfg = _make_awada_cfg(tmp_path, epochs=5)
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": {"save_every": 2}}))
        with (
            patch("tools.train_awada.AWADA", return_value=_make_full_mock_model()),
            patch("tools.train_awada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B, att_A, att_B, with_attention=True)),
            patch("tools.train_awada.AttentionPairedDataset"),
        ):
            awada_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("awada_epoch_")}
        assert saved == {"awada_epoch_2.pth", "awada_epoch_4.pth", "awada_epoch_5.pth"}

    def test_awada_default_save_every_saves_only_final_epoch(self, tmp_path):
        real_A = torch.zeros(1, 3, 64, 64)
        real_B = torch.zeros(1, 3, 64, 64)
        att_A = torch.zeros(1, 1, 64, 64)
        att_B = torch.zeros(1, 1, 64, 64)
        from tools.train_awada import _train as awada_train
        cfg = _make_awada_cfg(tmp_path, epochs=3)
        with (
            patch("tools.train_awada.AWADA", return_value=_make_full_mock_model()),
            patch("tools.train_awada.DataLoader", return_value=self._make_dataloader_mock(real_A, real_B, att_A, att_B, with_attention=True)),
            patch("tools.train_awada.AttentionPairedDataset"),
        ):
            awada_train(cfg)
        saved = {f for f in os.listdir(str(tmp_path)) if f.startswith("awada_epoch_")}
        assert saved == {"awada_epoch_3.pth"}

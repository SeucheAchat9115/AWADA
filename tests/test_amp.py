"""Tests for Automatic Mixed Precision (AMP) support in training scripts."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from awada.models.awada import AWADA
from awada.models.cyclegan import CycleGAN

# Path to configs directory
CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

# ---------------------------------------------------------------------------
# Hydra config defaults tests
# ---------------------------------------------------------------------------


def _get_cfg(config_name: str, overrides: list = None):
    """Helper to compose a Hydra config for testing."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    GlobalHydra.instance().clear()
    return cfg


@pytest.mark.parametrize(
    "config_name",
    [
        "train_cyclegan",
        "train_cycada",
        "train_awada",
        "train_detector",
    ],
)
def test_amp_defaults_to_false(config_name):
    """training.amp (or detector.amp) defaults to False when not supplied."""
    cfg = _get_cfg(config_name)
    if config_name == "train_detector":
        assert cfg.detector.amp is False
    else:
        assert cfg.training.amp is False


@pytest.mark.parametrize(
    "config_name",
    [
        "train_cyclegan",
        "train_cycada",
        "train_awada",
        "train_detector",
    ],
)
def test_amp_can_be_set_to_true(config_name):
    """Passing training.amp=true (or detector.amp=true) sets the flag to True."""
    if config_name == "train_detector":
        cfg = _get_cfg(config_name, overrides=["detector.amp=true"])
        assert cfg.detector.amp is True
    else:
        cfg = _get_cfg(config_name, overrides=["training.amp=true"])
        assert cfg.training.amp is True


# ---------------------------------------------------------------------------
# AMP code-path smoke tests (CPU, amp disabled so GradScaler is a no-op)
# ---------------------------------------------------------------------------

DEVICE = "cpu"
IMG_SIZE = 32
BATCH_SIZE = 1


def _make_gan_tensors():
    real_A = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    real_B = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    return real_A, real_B


def test_cyclegan_amp_code_path():
    """CycleGAN training step runs without error with AMP code path (disabled on CPU)."""
    model = CycleGAN(device=DEVICE)
    opt_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=2e-4)
    opt_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=2e-4)
    scaler_G = torch.cuda.amp.GradScaler(enabled=False)
    scaler_D = torch.cuda.amp.GradScaler(enabled=False)

    real_A, real_B = _make_gan_tensors()

    with torch.cuda.amp.autocast(enabled=False):
        model.set_input(real_A, real_B)
        model.forward()

    for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
        p.requires_grad_(False)
    opt_G.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        g_losses = model.compute_generator_loss()
    scaler_G.scale(g_losses["total_G"]).backward()
    scaler_G.step(opt_G)
    scaler_G.update()

    for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
        p.requires_grad_(True)
    opt_D.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        d_losses = model.compute_discriminator_loss()
    scaler_D.scale(d_losses["total_D"]).backward()
    scaler_D.step(opt_D)
    scaler_D.update()

    assert torch.isfinite(g_losses["total_G"])
    assert torch.isfinite(d_losses["total_D"])


def test_awada_amp_code_path():
    """AWADA training step runs without error with AMP code path (disabled on CPU)."""
    model = AWADA(device=DEVICE)
    opt_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=2e-4)
    opt_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=2e-4)
    scaler_G = torch.cuda.amp.GradScaler(enabled=False)
    scaler_D = torch.cuda.amp.GradScaler(enabled=False)

    real_A, real_B = _make_gan_tensors()
    att_A = torch.ones(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    att_B = torch.ones(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    with torch.cuda.amp.autocast(enabled=False):
        model.set_input(real_A, real_B, attention_A=att_A, attention_B=att_B)
        model.forward()

    for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
        p.requires_grad_(False)
    opt_G.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        g_losses = model.compute_generator_loss()
    scaler_G.scale(g_losses["total_G"]).backward()
    scaler_G.step(opt_G)
    scaler_G.update()

    for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
        p.requires_grad_(True)
    opt_D.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        d_losses = model.compute_discriminator_loss()
    scaler_D.scale(d_losses["total_D"]).backward()
    scaler_D.step(opt_D)
    scaler_D.update()

    assert torch.isfinite(g_losses["total_G"])
    assert torch.isfinite(d_losses["total_D"])


def test_gradscaler_state_dict_round_trip():
    """GradScaler state_dict can be saved and loaded (simulates checkpoint persistence)."""
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    state = scaler.state_dict()
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)
    scaler2.load_state_dict(state)
    assert scaler2.state_dict() == state

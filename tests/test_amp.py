"""Tests for Automatic Mixed Precision (AMP) support in training scripts."""

import argparse
from unittest.mock import patch

import pytest
import torch

from awada.models.awada import AWADA
from awada.models.cyclegan import CycleGAN

# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


def _parse_train_args(script_module_name, extra_argv):
    """Import the script module and invoke its parser with synthetic argv."""
    import importlib

    mod = importlib.import_module(script_module_name)
    # Reach into the module to extract the parser by temporarily patching
    # parse_args so we can capture the namespace without running main().
    captured = {}

    original_parse_args = argparse.ArgumentParser.parse_args

    def fake_parse_args(self, args=None, namespace=None):
        ns = original_parse_args(self, args=extra_argv, namespace=namespace)
        captured["ns"] = ns
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "parse_args", fake_parse_args):
        try:
            mod.main()
        except SystemExit:
            pass

    return captured.get("ns")


@pytest.mark.parametrize(
    "module",
    [
        "tools.train_cyclegan",
        "tools.train_cycada",
        "tools.train_awada",
        "tools.train_detector",
    ],
)
def test_amp_flag_defaults_to_false(module):
    """--amp defaults to False when not supplied."""
    # Provide the minimum required positional args so argparse doesn't error
    # before we capture the namespace.
    if module == "tools.train_detector":
        argv = [
            "--dataset",
            "sim10k",
            "--data_root",
            "/tmp",
            "--num_classes",
            "1",
            "--output_dir",
            "/tmp",
        ]
    elif module == "tools.train_awada":
        argv = [
            "--source_dir",
            "/tmp",
            "--target_dir",
            "/tmp",
            "--source_attention_dir",
            "/tmp",
            "--target_attention_dir",
            "/tmp",
            "--output_dir",
            "/tmp",
        ]
    else:
        argv = ["--source_dir", "/tmp", "--target_dir", "/tmp", "--output_dir", "/tmp"]

    ns = _parse_train_args(module, argv)
    assert ns is not None, f"Could not capture namespace from {module}"
    assert ns.amp is False


@pytest.mark.parametrize(
    "module",
    [
        "tools.train_cyclegan",
        "tools.train_cycada",
        "tools.train_awada",
        "tools.train_detector",
    ],
)
def test_amp_flag_set_to_true(module):
    """Passing --amp sets the flag to True."""
    if module == "tools.train_detector":
        argv = [
            "--dataset",
            "sim10k",
            "--data_root",
            "/tmp",
            "--num_classes",
            "1",
            "--output_dir",
            "/tmp",
            "--amp",
        ]
    elif module == "tools.train_awada":
        argv = [
            "--source_dir",
            "/tmp",
            "--target_dir",
            "/tmp",
            "--source_attention_dir",
            "/tmp",
            "--target_attention_dir",
            "/tmp",
            "--output_dir",
            "/tmp",
            "--amp",
        ]
    else:
        argv = ["--source_dir", "/tmp", "--target_dir", "/tmp", "--output_dir", "/tmp", "--amp"]

    ns = _parse_train_args(module, argv)
    assert ns is not None, f"Could not capture namespace from {module}"
    assert ns.amp is True


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

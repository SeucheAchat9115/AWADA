"""Tests for YAML configuration structure and value validation."""

import os

import pytest
import yaml

AWADA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "awada.yaml")
CYCADA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "cycada.yaml")
CYCLEGAN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "cyclegan.yaml")

_REQUIRED_KEYS = (
    "epochs",
    "lr",
    "betas",
    "lambda_gan",
    "lambda_cyc",
    "lambda_idt",
    "lambda_sem",
    "batch_size",
    "patch_size",
    "resize_min_side",
    "device",
)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def awada_config():
    return _load(AWADA_CONFIG_PATH)


@pytest.fixture(scope="module")
def cycada_config():
    return _load(CYCADA_CONFIG_PATH)


@pytest.fixture(scope="module")
def cyclegan_config():
    return _load(CYCLEGAN_CONFIG_PATH)


class TestAwadaConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(AWADA_CONFIG_PATH)

    def test_required_keys_present(self, awada_config):
        for key in _REQUIRED_KEYS:
            assert key in awada_config, f"Missing config key: {key}"

    def test_epochs_positive(self, awada_config):
        assert awada_config["epochs"] > 0

    def test_lr_positive(self, awada_config):
        assert awada_config["lr"] > 0

    def test_betas_has_two_elements(self, awada_config):
        assert len(awada_config["betas"]) == 2

    def test_betas_in_unit_interval(self, awada_config):
        for b in awada_config["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, awada_config):
        assert awada_config["lambda_gan"] > 0
        assert awada_config["lambda_cyc"] > 0

    def test_lambda_optional_weights_non_negative(self, awada_config):
        """lambda_idt and lambda_sem default to 0 (disabled) and must be >= 0."""
        assert awada_config["lambda_idt"] >= 0
        assert awada_config["lambda_sem"] >= 0

    def test_batch_size_positive(self, awada_config):
        assert awada_config["batch_size"] >= 1

    def test_patch_size_positive(self, awada_config):
        assert awada_config["patch_size"] >= 1

    def test_resize_min_side_positive(self, awada_config):
        assert awada_config["resize_min_side"] > 0

    def test_device_is_string(self, awada_config):
        assert isinstance(awada_config["device"], str)


class TestCyCadaConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(CYCADA_CONFIG_PATH)

    def test_required_keys_present(self, cycada_config):
        for key in _REQUIRED_KEYS:
            assert key in cycada_config, f"Missing config key: {key}"

    def test_epochs_positive(self, cycada_config):
        assert cycada_config["epochs"] > 0

    def test_lr_positive(self, cycada_config):
        assert cycada_config["lr"] > 0

    def test_betas_has_two_elements(self, cycada_config):
        assert len(cycada_config["betas"]) == 2

    def test_betas_in_unit_interval(self, cycada_config):
        for b in cycada_config["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, cycada_config):
        assert cycada_config["lambda_gan"] > 0
        assert cycada_config["lambda_cyc"] > 0

    def test_lambda_sem_enabled(self, cycada_config):
        """CyCada config must have lambda_sem > 0 to enable semantic consistency loss."""
        assert cycada_config["lambda_sem"] > 0

    def test_lambda_idt_non_negative(self, cycada_config):
        assert cycada_config["lambda_idt"] >= 0

    def test_batch_size_positive(self, cycada_config):
        assert cycada_config["batch_size"] >= 1

    def test_patch_size_positive(self, cycada_config):
        assert cycada_config["patch_size"] >= 1

    def test_resize_min_side_positive(self, cycada_config):
        assert cycada_config["resize_min_side"] > 0

    def test_device_is_string(self, cycada_config):
        assert isinstance(cycada_config["device"], str)


class TestCycleGANConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(CYCLEGAN_CONFIG_PATH)

    def test_required_keys_present(self, cyclegan_config):
        for key in _REQUIRED_KEYS:
            assert key in cyclegan_config, f"Missing config key: {key}"

    def test_epochs_positive(self, cyclegan_config):
        assert cyclegan_config["epochs"] > 0

    def test_lr_positive(self, cyclegan_config):
        assert cyclegan_config["lr"] > 0

    def test_betas_has_two_elements(self, cyclegan_config):
        assert len(cyclegan_config["betas"]) == 2

    def test_betas_in_unit_interval(self, cyclegan_config):
        for b in cyclegan_config["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, cyclegan_config):
        assert cyclegan_config["lambda_gan"] > 0
        assert cyclegan_config["lambda_cyc"] > 0

    def test_lambda_optional_weights_non_negative(self, cyclegan_config):
        assert cyclegan_config["lambda_idt"] >= 0
        assert cyclegan_config["lambda_sem"] >= 0

    def test_batch_size_positive(self, cyclegan_config):
        assert cyclegan_config["batch_size"] >= 1

    def test_patch_size_positive(self, cyclegan_config):
        assert cyclegan_config["patch_size"] >= 1

    def test_resize_min_side_positive(self, cyclegan_config):
        assert cyclegan_config["resize_min_side"] > 0

    def test_device_is_string(self, cyclegan_config):
        assert isinstance(cyclegan_config["device"], str)

"""Tests for Hydra YAML configuration structure and value validation.

Configs are now split across config groups:
- configs/train_<model>.yaml  → training and data sections
- configs/model/<model>.yaml  → loss weights and buffer settings
- configs/hardware/default.yaml → device setting
"""

import os

import pytest
import yaml

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

TRAIN_AWADA_PATH = os.path.join(_CONFIGS_DIR, "train_awada.yaml")
TRAIN_CYCADA_PATH = os.path.join(_CONFIGS_DIR, "train_cycada.yaml")
TRAIN_CYCLEGAN_PATH = os.path.join(_CONFIGS_DIR, "train_cyclegan.yaml")

MODEL_AWADA_PATH = os.path.join(_CONFIGS_DIR, "model", "awada.yaml")
MODEL_CYCADA_PATH = os.path.join(_CONFIGS_DIR, "model", "cycada.yaml")
MODEL_CYCLEGAN_PATH = os.path.join(_CONFIGS_DIR, "model", "cyclegan.yaml")

HARDWARE_DEFAULT_PATH = os.path.join(_CONFIGS_DIR, "hardware", "default.yaml")

_REQUIRED_TRAINING_KEYS = ("epochs", "lr", "betas", "batch_size", "patch_size", "resize_min_side")
_REQUIRED_MODEL_KEYS = ("lambda_gan", "lambda_cyc", "lambda_idt", "lambda_sem")


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def awada_train():
    return _load(TRAIN_AWADA_PATH)["training"]


@pytest.fixture(scope="module")
def awada_model():
    return _load(MODEL_AWADA_PATH)


@pytest.fixture(scope="module")
def cycada_train():
    return _load(TRAIN_CYCADA_PATH)["training"]


@pytest.fixture(scope="module")
def cycada_model():
    return _load(MODEL_CYCADA_PATH)


@pytest.fixture(scope="module")
def cyclegan_train():
    return _load(TRAIN_CYCLEGAN_PATH)["training"]


@pytest.fixture(scope="module")
def cyclegan_model():
    return _load(MODEL_CYCLEGAN_PATH)


@pytest.fixture(scope="module")
def hardware_config():
    return _load(HARDWARE_DEFAULT_PATH)


class TestAwadaConfig:
    def test_train_config_file_exists(self):
        assert os.path.isfile(TRAIN_AWADA_PATH)

    def test_model_config_file_exists(self):
        assert os.path.isfile(MODEL_AWADA_PATH)

    def test_required_training_keys_present(self, awada_train):
        for key in _REQUIRED_TRAINING_KEYS:
            assert key in awada_train, f"Missing training key: {key}"

    def test_required_model_keys_present(self, awada_model):
        for key in _REQUIRED_MODEL_KEYS:
            assert key in awada_model, f"Missing model key: {key}"

    def test_epochs_positive(self, awada_train):
        assert awada_train["epochs"] > 0

    def test_lr_positive(self, awada_train):
        assert awada_train["lr"] > 0

    def test_betas_has_two_elements(self, awada_train):
        assert len(awada_train["betas"]) == 2

    def test_betas_in_unit_interval(self, awada_train):
        for b in awada_train["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, awada_model):
        assert awada_model["lambda_gan"] > 0
        assert awada_model["lambda_cyc"] > 0

    def test_lambda_optional_weights_non_negative(self, awada_model):
        """lambda_idt and lambda_sem default to 0 (disabled) and must be >= 0."""
        assert awada_model["lambda_idt"] >= 0
        assert awada_model["lambda_sem"] >= 0

    def test_batch_size_positive(self, awada_train):
        assert awada_train["batch_size"] >= 1

    def test_patch_size_positive(self, awada_train):
        assert awada_train["patch_size"] >= 1

    def test_resize_min_side_positive(self, awada_train):
        assert awada_train["resize_min_side"] > 0


class TestCyCadaConfig:
    def test_train_config_file_exists(self):
        assert os.path.isfile(TRAIN_CYCADA_PATH)

    def test_model_config_file_exists(self):
        assert os.path.isfile(MODEL_CYCADA_PATH)

    def test_required_training_keys_present(self, cycada_train):
        for key in _REQUIRED_TRAINING_KEYS:
            assert key in cycada_train, f"Missing training key: {key}"

    def test_required_model_keys_present(self, cycada_model):
        for key in _REQUIRED_MODEL_KEYS:
            assert key in cycada_model, f"Missing model key: {key}"

    def test_epochs_positive(self, cycada_train):
        assert cycada_train["epochs"] > 0

    def test_lr_positive(self, cycada_train):
        assert cycada_train["lr"] > 0

    def test_betas_has_two_elements(self, cycada_train):
        assert len(cycada_train["betas"]) == 2

    def test_betas_in_unit_interval(self, cycada_train):
        for b in cycada_train["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, cycada_model):
        assert cycada_model["lambda_gan"] > 0
        assert cycada_model["lambda_cyc"] > 0

    def test_lambda_sem_enabled(self, cycada_model):
        """CyCada config must have lambda_sem > 0 to enable semantic consistency loss."""
        assert cycada_model["lambda_sem"] > 0

    def test_lambda_idt_non_negative(self, cycada_model):
        assert cycada_model["lambda_idt"] >= 0

    def test_batch_size_positive(self, cycada_train):
        assert cycada_train["batch_size"] >= 1

    def test_patch_size_positive(self, cycada_train):
        assert cycada_train["patch_size"] >= 1

    def test_resize_min_side_positive(self, cycada_train):
        assert cycada_train["resize_min_side"] > 0


class TestCycleGANConfig:
    def test_train_config_file_exists(self):
        assert os.path.isfile(TRAIN_CYCLEGAN_PATH)

    def test_model_config_file_exists(self):
        assert os.path.isfile(MODEL_CYCLEGAN_PATH)

    def test_required_training_keys_present(self, cyclegan_train):
        for key in _REQUIRED_TRAINING_KEYS:
            assert key in cyclegan_train, f"Missing training key: {key}"

    def test_required_model_keys_present(self, cyclegan_model):
        for key in _REQUIRED_MODEL_KEYS:
            assert key in cyclegan_model, f"Missing model key: {key}"

    def test_epochs_positive(self, cyclegan_train):
        assert cyclegan_train["epochs"] > 0

    def test_lr_positive(self, cyclegan_train):
        assert cyclegan_train["lr"] > 0

    def test_betas_has_two_elements(self, cyclegan_train):
        assert len(cyclegan_train["betas"]) == 2

    def test_betas_in_unit_interval(self, cyclegan_train):
        for b in cyclegan_train["betas"]:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, cyclegan_model):
        assert cyclegan_model["lambda_gan"] > 0
        assert cyclegan_model["lambda_cyc"] > 0

    def test_lambda_optional_weights_non_negative(self, cyclegan_model):
        assert cyclegan_model["lambda_idt"] >= 0
        assert cyclegan_model["lambda_sem"] >= 0

    def test_batch_size_positive(self, cyclegan_train):
        assert cyclegan_train["batch_size"] >= 1

    def test_patch_size_positive(self, cyclegan_train):
        assert cyclegan_train["patch_size"] >= 1

    def test_resize_min_side_positive(self, cyclegan_train):
        assert cyclegan_train["resize_min_side"] > 0


class TestHardwareConfig:
    def test_hardware_config_file_exists(self):
        assert os.path.isfile(HARDWARE_DEFAULT_PATH)

    def test_device_is_string(self, hardware_config):
        assert isinstance(hardware_config["device"], str)

    def test_device_non_empty(self, hardware_config):
        assert hardware_config["device"]

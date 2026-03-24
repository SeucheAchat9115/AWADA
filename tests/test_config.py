"""Tests for awada.yaml configuration structure and value validation."""
import os

import pytest
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'awada.yaml'
)


@pytest.fixture(scope='module')
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class TestAwadaConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(CONFIG_PATH)

    def test_required_keys_present(self, config):
        required = ('epochs', 'lr', 'betas', 'lambda_gan', 'lambda_cyc',
                    'lambda_idt', 'batch_size', 'patch_size', 'device')
        for key in required:
            assert key in config, f"Missing config key: {key}"

    def test_epochs_positive(self, config):
        assert config['epochs'] > 0

    def test_lr_positive(self, config):
        assert config['lr'] > 0

    def test_betas_has_two_elements(self, config):
        assert len(config['betas']) == 2

    def test_betas_in_unit_interval(self, config):
        for b in config['betas']:
            assert 0.0 <= b < 1.0

    def test_lambda_weights_positive(self, config):
        assert config['lambda_gan'] > 0
        assert config['lambda_cyc'] > 0
        assert config['lambda_idt'] > 0

    def test_batch_size_positive(self, config):
        assert config['batch_size'] >= 1

    def test_patch_size_positive(self, config):
        assert config['patch_size'] >= 1

    def test_device_is_string(self, config):
        assert isinstance(config['device'], str)

"""Tests for YAML configuration structure and value validation."""

import os

import pytest
import yaml

AWADA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "awada.yaml")
CYCADA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "cycada.yaml")
CYCLEGAN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "cyclegan.yaml")

CITYSCAPES_DATASET_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "datasets", "cityscapes.yaml"
)
BDD100K_DATASET_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "datasets", "bdd100k.yaml"
)
SIM10K_DATASET_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "datasets", "sim10k.yaml"
)
NORMALIZATION_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "datasets", "normalization.yaml"
)

_BENCHMARK_CONFIG_PATHS = {
    "sim10k_to_cityscapes": os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "benchmarks",
        "sim10k_to_cityscapes.yaml",
    ),
    "cityscapes_to_foggy": os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "benchmarks",
        "cityscapes_to_foggy.yaml",
    ),
    "cityscapes_to_bdd100k": os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "benchmarks",
        "cityscapes_to_bdd100k.yaml",
    ),
}

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

_REQUIRED_BENCHMARK_KEYS = (
    "benchmark",
    "source_dataset",
    "source_root",
    "source_images",
    "target_dataset",
    "target_root",
    "target_images",
    "num_classes",
    "output_suffix",
    "detector_epochs",
    "detector_batch_size",
    "detector_lr",
    "detector_pretrained",
    "score_threshold",
    "attention_split",
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


@pytest.fixture(scope="module")
def cityscapes_dataset_config():
    return _load(CITYSCAPES_DATASET_CONFIG_PATH)


@pytest.fixture(scope="module")
def bdd100k_dataset_config():
    return _load(BDD100K_DATASET_CONFIG_PATH)


@pytest.fixture(scope="module")
def sim10k_dataset_config():
    return _load(SIM10K_DATASET_CONFIG_PATH)


@pytest.fixture(scope="module")
def normalization_config():
    return _load(NORMALIZATION_CONFIG_PATH)


@pytest.fixture(params=list(_BENCHMARK_CONFIG_PATHS.keys()), scope="module")
def benchmark_config(request):
    return _load(_BENCHMARK_CONFIG_PATHS[request.param])


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


class TestCityscapesDatasetConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(CITYSCAPES_DATASET_CONFIG_PATH)

    def test_label_map_is_dict(self, cityscapes_dataset_config):
        assert isinstance(cityscapes_dataset_config["label_map"], dict)

    def test_label_map_has_eight_classes(self, cityscapes_dataset_config):
        assert len(cityscapes_dataset_config["label_map"]) == 8

    def test_class_names_has_eight_entries(self, cityscapes_dataset_config):
        assert len(cityscapes_dataset_config["class_names"]) == 8

    def test_cityscapes_bdd100k_label_map_has_seven_classes(self, cityscapes_dataset_config):
        assert len(cityscapes_dataset_config["bdd100k_label_map"]) == 7

    def test_cityscapes_bdd100k_aligned_classes_has_seven_entries(self, cityscapes_dataset_config):
        assert len(cityscapes_dataset_config["bdd100k_aligned_classes"]) == 7

    def test_min_pixels_threshold_positive(self, cityscapes_dataset_config):
        assert cityscapes_dataset_config["min_pixels_threshold"] > 0

    def test_min_box_dim_positive(self, cityscapes_dataset_config):
        assert cityscapes_dataset_config["min_box_dim"] > 0

    def test_label_map_values_are_one_based(self, cityscapes_dataset_config):
        values = list(cityscapes_dataset_config["label_map"].values())
        assert min(values) == 1

    def test_bdd100k_label_map_values_are_one_based(self, cityscapes_dataset_config):
        values = list(cityscapes_dataset_config["bdd100k_label_map"].values())
        assert min(values) == 1


class TestBdd100kDatasetConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(BDD100K_DATASET_CONFIG_PATH)

    def test_label_map_is_dict(self, bdd100k_dataset_config):
        assert isinstance(bdd100k_dataset_config["label_map"], dict)

    def test_label_map_has_seven_classes(self, bdd100k_dataset_config):
        assert len(bdd100k_dataset_config["label_map"]) == 7

    def test_class_names_has_seven_entries(self, bdd100k_dataset_config):
        assert len(bdd100k_dataset_config["class_names"]) == 7

    def test_min_box_dim_positive(self, bdd100k_dataset_config):
        assert bdd100k_dataset_config["min_box_dim"] > 0

    def test_label_map_values_are_one_based(self, bdd100k_dataset_config):
        values = list(bdd100k_dataset_config["label_map"].values())
        assert min(values) == 1


class TestSim10kDatasetConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(SIM10K_DATASET_CONFIG_PATH)

    def test_class_names_contains_car(self, sim10k_dataset_config):
        assert "car" in sim10k_dataset_config["class_names"]

    def test_min_box_dim_positive(self, sim10k_dataset_config):
        assert sim10k_dataset_config["min_box_dim"] > 0


class TestNormalizationConfig:
    def test_config_file_exists(self):
        assert os.path.isfile(NORMALIZATION_CONFIG_PATH)

    def test_mean_has_three_channels(self, normalization_config):
        assert len(normalization_config["mean"]) == 3

    def test_std_has_three_channels(self, normalization_config):
        assert len(normalization_config["std"]) == 3

    def test_mean_values_in_unit_interval(self, normalization_config):
        for v in normalization_config["mean"]:
            assert 0.0 <= v <= 1.0

    def test_std_values_positive(self, normalization_config):
        for v in normalization_config["std"]:
            assert v > 0.0


class TestBenchmarkConfigs:
    def test_required_keys_present(self, benchmark_config):
        for key in _REQUIRED_BENCHMARK_KEYS:
            assert key in benchmark_config, f"Missing benchmark config key: {key}"

    def test_benchmark_name_is_string(self, benchmark_config):
        assert isinstance(benchmark_config["benchmark"], str)

    def test_num_classes_positive(self, benchmark_config):
        assert benchmark_config["num_classes"] >= 1

    def test_detector_epochs_positive(self, benchmark_config):
        assert benchmark_config["detector_epochs"] >= 1

    def test_detector_batch_size_positive(self, benchmark_config):
        assert benchmark_config["detector_batch_size"] >= 1

    def test_detector_lr_positive(self, benchmark_config):
        assert benchmark_config["detector_lr"] > 0

    def test_detector_pretrained_is_bool(self, benchmark_config):
        assert isinstance(benchmark_config["detector_pretrained"], bool)

    def test_score_threshold_in_unit_interval(self, benchmark_config):
        assert 0.0 <= benchmark_config["score_threshold"] <= 1.0

    def test_attention_split_is_string(self, benchmark_config):
        assert isinstance(benchmark_config["attention_split"], str)

    def test_output_suffix_is_string(self, benchmark_config):
        assert isinstance(benchmark_config["output_suffix"], str)
        assert len(benchmark_config["output_suffix"]) > 0


class TestConfigPyConstants:
    """Verify that awada.config still exports the expected Python constants."""

    def test_default_device_is_string(self):
        from awada.config import DEFAULT_DEVICE

        assert isinstance(DEFAULT_DEVICE, str)
        assert DEFAULT_DEVICE in ("cuda", "cpu")

    def test_imagenet_mean_has_three_channels(self):
        from awada.config import IMAGENET_MEAN

        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_has_three_channels(self):
        from awada.config import IMAGENET_STD

        assert len(IMAGENET_STD) == 3

    def test_cityscapes_label_map_keys_are_ints(self):
        from awada.config import CITYSCAPES_LABEL_MAP

        for k in CITYSCAPES_LABEL_MAP:
            assert isinstance(k, int)

    def test_cityscapes_label_map_has_eight_classes(self):
        from awada.config import CITYSCAPES_LABEL_MAP

        assert len(CITYSCAPES_LABEL_MAP) == 8

    def test_cityscapes_bdd100k_label_map_has_seven_classes(self):
        from awada.config import CITYSCAPES_BDD100K_LABEL_MAP

        assert len(CITYSCAPES_BDD100K_LABEL_MAP) == 7

    def test_bdd100k_label_map_keys_are_strings(self):
        from awada.config import BDD100K_LABEL_MAP

        for k in BDD100K_LABEL_MAP:
            assert isinstance(k, str)

    def test_min_box_dim_positive(self):
        from awada.config import MIN_BOX_DIM

        assert MIN_BOX_DIM > 0

    def test_min_pixels_threshold_positive(self):
        from awada.config import MIN_PIXELS_THRESHOLD

        assert MIN_PIXELS_THRESHOLD > 0

    def test_sim10k_class_names_contains_car(self):
        from awada.config import SIM10K_CLASS_NAMES

        assert "car" in SIM10K_CLASS_NAMES

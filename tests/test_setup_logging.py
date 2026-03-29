"""Tests for the setup_logging utility in awada.utils.train_utils."""

import logging
import os
from datetime import date

import pytest

from awada.utils.train_utils import setup_logging


@pytest.fixture(autouse=True)
def _cleanup_file_handlers():
    """Remove any FileHandlers added during the test so handlers don't accumulate."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for h in list(root.handlers):
        if h not in before:
            h.flush()
            h.close()
            root.removeHandler(h)


def test_setup_logging_returns_dated_dir(tmp_path):
    """Return value must be <output_dir>/<today ISO>."""
    dated_dir = setup_logging(str(tmp_path))
    assert dated_dir == os.path.join(str(tmp_path), date.today().isoformat())


def test_setup_logging_creates_dated_dir(tmp_path):
    """The dated sub-directory must exist after the call."""
    dated_dir = setup_logging(str(tmp_path))
    assert os.path.isdir(dated_dir)


def test_setup_logging_creates_log_file(tmp_path):
    """A ``training.log`` file must be created inside the dated directory."""
    dated_dir = setup_logging(str(tmp_path))
    log_path = os.path.join(dated_dir, "training.log")
    assert os.path.isfile(log_path)


def test_setup_logging_writes_messages(tmp_path):
    """Log records must appear in the file after the call."""
    # Ensure the root logger passes INFO records to handlers (pytest may have
    # already installed its own handler, making basicConfig() a no-op).
    root = logging.getLogger()
    original_level = root.level
    root.setLevel(logging.INFO)
    try:
        dated_dir = setup_logging(str(tmp_path))
        log_path = os.path.join(dated_dir, "training.log")

        test_logger = logging.getLogger("test_setup_logging_writes_messages")
        test_logger.info("sentinel message abc123")

        for handler in root.handlers:
            handler.flush()

        content = open(log_path).read()
        assert "sentinel message abc123" in content
    finally:
        root.setLevel(original_level)


def test_setup_logging_custom_filename(tmp_path):
    """Custom ``log_filename`` must be used instead of the default."""
    dated_dir = setup_logging(str(tmp_path), log_filename="experiment.log")
    log_path = os.path.join(dated_dir, "experiment.log")
    assert os.path.isfile(log_path)


def test_setup_logging_idempotent_dir(tmp_path):
    """Calling setup_logging twice must not raise even if the dated dir already exists."""
    setup_logging(str(tmp_path))
    setup_logging(str(tmp_path))  # must not raise

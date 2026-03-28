"""Tests for the set_seed reproducibility helper."""

import random

import numpy as np
import torch

from awada.utils.train_utils import set_seed


def _sample_values(seed):
    """Call set_seed then draw one value from each RNG."""
    set_seed(seed)
    r = random.random()
    n = np.random.rand()
    t = torch.rand(1).item()
    return r, n, t


class TestSetSeed:
    def test_default_seed_is_42(self):
        """set_seed() with no argument should use seed=42."""
        set_seed()
        r1 = random.random()
        set_seed(42)
        r2 = random.random()
        assert r1 == r2

    def test_same_seed_gives_same_values(self):
        """Two calls with the same seed must produce identical RNG outputs."""
        v1 = _sample_values(0)
        v2 = _sample_values(0)
        assert v1 == v2

    def test_different_seeds_give_different_values(self):
        """Different seeds must not produce identical outputs (with overwhelming probability)."""
        v1 = _sample_values(1)
        v2 = _sample_values(2)
        assert v1 != v2

    def test_torch_manual_seed_applied(self):
        set_seed(7)
        t1 = torch.rand(4)
        set_seed(7)
        t2 = torch.rand(4)
        assert torch.equal(t1, t2)

    def test_numpy_seed_applied(self):
        set_seed(99)
        n1 = np.random.rand(4)
        set_seed(99)
        n2 = np.random.rand(4)
        assert np.array_equal(n1, n2)

    def test_random_seed_applied(self):
        set_seed(123)
        r1 = [random.random() for _ in range(4)]
        set_seed(123)
        r2 = [random.random() for _ in range(4)]
        assert r1 == r2

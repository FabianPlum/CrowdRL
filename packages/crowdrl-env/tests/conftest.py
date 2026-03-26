"""Shared fixtures for crowdrl-env tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)

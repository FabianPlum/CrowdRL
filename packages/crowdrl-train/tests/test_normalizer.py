"""Tests for observation and reward normalization."""

from __future__ import annotations

import numpy as np
import pytest

from crowdrl_train.normalizer import RewardNormalizer, RunningNormalizer


class TestRunningNormalizer:
    def test_converges_to_known_distribution(self):
        """Running stats should converge to true mean/var on known data."""
        rng = np.random.default_rng(42)
        normalizer = RunningNormalizer(shape=(3,))

        true_mean = np.array([1.0, -2.0, 5.0])
        true_std = np.array([0.5, 1.0, 2.0])

        for _ in range(100):
            batch = rng.normal(true_mean, true_std, size=(50, 3))
            normalizer.update(batch)

        np.testing.assert_allclose(normalizer.mean, true_mean, atol=0.1)
        np.testing.assert_allclose(np.sqrt(normalizer.var), true_std, atol=0.15)

    def test_normalize_zero_mean_unit_var(self):
        """Normalised output should be approximately zero-mean, unit-var."""
        rng = np.random.default_rng(42)
        normalizer = RunningNormalizer(shape=(5,))

        data = rng.normal(loc=10.0, scale=3.0, size=(1000, 5))
        normalizer.update(data)

        normed = normalizer.normalize(data)
        assert abs(normed.mean()) < 0.1
        assert abs(normed.std() - 1.0) < 0.1

    def test_clip_bounds(self):
        """Normalised values should be clipped to [-clip, clip]."""
        normalizer = RunningNormalizer(shape=(1,), clip=5.0)
        normalizer.update(np.array([[0.0]]))  # mean=0, var≈0

        extreme = np.array([[1000.0]])
        normed = normalizer.normalize(extreme)
        assert normed[0, 0] == pytest.approx(5.0, abs=0.1)

    def test_state_dict_roundtrip(self):
        """State dict should preserve statistics exactly."""
        rng = np.random.default_rng(42)
        normalizer = RunningNormalizer(shape=(3,))
        normalizer.update(rng.normal(size=(100, 3)))

        state = normalizer.state_dict()
        restored = RunningNormalizer(shape=(3,))
        restored.load_state_dict(state)

        np.testing.assert_array_equal(normalizer.mean, restored.mean)
        np.testing.assert_array_equal(normalizer.var, restored.var)
        assert normalizer.count == restored.count

    def test_single_sample_update(self):
        """Should handle single-sample updates without error."""
        normalizer = RunningNormalizer(shape=(2,))
        normalizer.update(np.array([1.0, 2.0]))
        assert normalizer.mean.shape == (2,)

    def test_incremental_matches_batch(self):
        """Incremental updates should give ~same result as one batch update."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(200, 4))

        batch_norm = RunningNormalizer(shape=(4,))
        batch_norm.update(data)

        incr_norm = RunningNormalizer(shape=(4,))
        for i in range(0, 200, 10):
            incr_norm.update(data[i : i + 10])

        np.testing.assert_allclose(batch_norm.mean, incr_norm.mean, atol=1e-10)
        np.testing.assert_allclose(batch_norm.var, incr_norm.var, atol=1e-6)


class TestRewardNormalizer:
    def test_normalizes_rewards(self):
        """Normalised rewards should have reduced variance."""
        rng = np.random.default_rng(42)
        normalizer = RewardNormalizer(gamma=0.99)

        # Feed some rewards to build up statistics
        for _ in range(50):
            rewards = rng.normal(loc=5.0, scale=2.0, size=(10,))
            dones = np.zeros(10, dtype=np.bool_)
            normed = normalizer.normalize(rewards, dones)

        # After warmup, normed rewards should be smaller in magnitude
        rewards = np.full(10, 5.0)
        normed = normalizer.normalize(rewards, np.zeros(10, dtype=np.bool_))
        assert np.abs(normed).mean() < np.abs(rewards).mean()

    def test_state_dict_roundtrip(self):
        rng = np.random.default_rng(42)
        normalizer = RewardNormalizer(gamma=0.99)
        for _ in range(20):
            normalizer.normalize(rng.normal(size=(5,)), np.zeros(5, dtype=np.bool_))

        state = normalizer.state_dict()
        restored = RewardNormalizer(gamma=0.99)
        restored.load_state_dict(state)

        assert normalizer._running_return == restored._running_return

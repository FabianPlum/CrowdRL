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

    def test_count_capped_no_overflow(self):
        """The DDP reward-normalizer sync re-sums the already-merged count every
        rollout, so an UNCAPPED count grows geometrically to ~1e305 -- then
        m_a = var * count overflows to inf and var -> NaN -> every normalized
        value NaN (the deterministic r360 collapse, twin of the obs-normalizer
        r355 overflow). The count must stay capped, even when restored from a
        checkpoint whose count already overflowed under the old bug."""
        from crowdrl_train.normalizer import _MAX_COUNT

        norm = RunningNormalizer(shape=(1,))
        norm.update(np.random.default_rng(0).normal(scale=3.0, size=(500, 1)))

        # A checkpoint / sync that drove the count past float64 sanity.
        st = norm.state_dict()
        st["count"] = 1e307  # var(~9) * 1e307 ~ 1e308 -> would overflow uncapped
        norm.load_state_dict(st)
        assert norm.count <= _MAX_COUNT, "load_state_dict must cap an overflowed count"

        # Subsequent updates stay finite and bounded.
        rng = np.random.default_rng(1)
        for _ in range(50):
            norm.update(rng.normal(scale=3.0, size=(200, 1)))
            assert norm.count <= _MAX_COUNT
        assert np.isfinite(norm.mean).all()
        assert np.isfinite(norm.var).all()
        assert np.isfinite(norm.normalize(np.array([[1.0]]))).all()

    def test_update_drops_nonfinite_samples(self):
        """A single NaN/Inf sample must not permanently poison running mean/var
        (which would NaN every future normalized value)."""
        norm = RunningNormalizer(shape=(2,))
        good = np.array([[1.0, 2.0], [1.0, 2.0]])
        bad = np.array([[np.nan, 0.0], [np.inf, 0.0]])
        norm.update(np.concatenate([good, bad], axis=0))
        assert np.isfinite(norm.mean).all()
        assert np.isfinite(norm.var).all()

        # An all-non-finite batch is a no-op (stats unchanged).
        m0, v0, c0 = norm.mean.copy(), norm.var.copy(), norm.count
        norm.update(np.array([[np.nan, np.inf]]))
        np.testing.assert_array_equal(norm.mean, m0)
        np.testing.assert_array_equal(norm.var, v0)
        assert norm.count == c0


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

    def test_overflowed_return_var_recovers_finite(self):
        """The r360 collapse: the internal return-variance tracker's count is
        DDP-synced (re-summed) every rollout, so without a cap it overflows and
        std = sqrt(var) -> NaN -> every normalized reward NaN -> value_loss NaN
        -> frozen policy. Even from an already-overflowed count, normalize() must
        return finite rewards (the cap rescues a poisoned checkpoint too)."""
        from crowdrl_train.normalizer import _MAX_COUNT

        norm = RewardNormalizer(gamma=0.99)
        rng = np.random.default_rng(7)
        for _ in range(30):
            norm.normalize(
                rng.normal(loc=-0.2, scale=1.0, size=(16,)), np.zeros(16, dtype=np.bool_)
            )

        # Simulate the uncapped DDP sync having driven the count to overflow.
        st = norm.state_dict()
        st["return_var"]["count"] = 1e307
        norm.load_state_dict(st)
        assert norm._return_var.count <= _MAX_COUNT

        out = norm.normalize(np.full(16, -0.2), np.zeros(16, dtype=np.bool_))
        assert np.isfinite(out).all(), "overflowed return-var must not NaN the rewards"

"""Tests for DDP helpers (single-process, no GPU required).

All distributed functions are no-ops when ``torch.distributed`` is not
initialised, which makes them safe to unit-test without ``torchrun``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from crowdrl_torch.distributed import (
    allreduce_gradients,
    broadcast_curriculum_state,
    cleanup_distributed,
    distributed_seed,
    gather_episode_stats,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_rank,
    seed_everything,
    sync_reward_normalizer,
)
from crowdrl_torch.normalizer import TorchRunningNormalizer
from crowdrl_train.normalizer import RewardNormalizer


# ---------------------------------------------------------------------------
# Rank queries default to single-process values
# ---------------------------------------------------------------------------


class TestRankQueries:
    def test_not_distributed_by_default(self):
        assert not is_distributed()

    def test_main_rank_when_not_distributed(self):
        assert is_main_rank()

    def test_rank_zero_when_not_distributed(self):
        assert get_rank() == 0

    def test_world_size_one_when_not_distributed(self):
        assert get_world_size() == 1


# ---------------------------------------------------------------------------
# allreduce_gradients is a no-op outside distributed
# ---------------------------------------------------------------------------


class TestAllreduceGradients:
    def test_noop_when_not_distributed(self):
        model = nn.Linear(4, 2)
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()

        # Snapshot grads before
        grads_before = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        allreduce_gradients(model)

        # Grads should be unchanged
        for name, param in model.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(param.grad, grads_before[name])

    def test_noop_with_no_grads(self):
        model = nn.Linear(4, 2)
        # No backward called -- no grads
        allreduce_gradients(model)  # should not raise


# ---------------------------------------------------------------------------
# sync_across_ranks on TorchRunningNormalizer
# ---------------------------------------------------------------------------


class TestNormalizerSync:
    def test_noop_when_not_distributed(self):
        norm = TorchRunningNormalizer((4,), device="cpu")
        norm.update(torch.randn(10, 4))

        mean_before = norm.mean.clone()
        var_before = norm.var.clone()
        count_before = norm.count

        norm.sync_across_ranks()

        torch.testing.assert_close(norm.mean, mean_before)
        torch.testing.assert_close(norm.var, var_before)
        assert norm.count == count_before


# ---------------------------------------------------------------------------
# sync_reward_normalizer
# ---------------------------------------------------------------------------


class TestRewardNormalizerSync:
    def test_noop_when_not_distributed(self):
        rn = RewardNormalizer(gamma=0.99)
        # Feed some data
        rn.normalize(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]))

        state_before = rn.state_dict()

        sync_reward_normalizer(rn, torch.device("cpu"))

        state_after = rn.state_dict()
        np.testing.assert_array_equal(
            state_before["return_var"]["mean"],
            state_after["return_var"]["mean"],
        )
        assert state_before["running_return"] == state_after["running_return"]


# ---------------------------------------------------------------------------
# gather_episode_stats
# ---------------------------------------------------------------------------


class TestGatherEpisodeStats:
    def test_passthrough_when_not_distributed(self):
        episodes = [
            {"goal_rate": 0.5, "mean_reward": 1.0},
            {"goal_rate": 0.8, "mean_reward": 2.0},
        ]
        result = gather_episode_stats(episodes)
        assert result is episodes

    def test_empty_list_passthrough(self):
        result = gather_episode_stats([])
        assert result == []


# ---------------------------------------------------------------------------
# broadcast_curriculum_state
# ---------------------------------------------------------------------------


class _FakeCurriculum:
    """Minimal stand-in with state_dict/load_state_dict."""

    def __init__(self, phase_idx: int = 0):
        self.phase_idx = phase_idx

    def state_dict(self) -> dict:
        return {"phase_idx": self.phase_idx}

    def load_state_dict(self, state: dict) -> None:
        self.phase_idx = state["phase_idx"]


class TestBroadcastCurriculum:
    def test_noop_when_not_distributed(self):
        cm = _FakeCurriculum(phase_idx=3)
        broadcast_curriculum_state(cm)
        assert cm.phase_idx == 3  # unchanged


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_distributed_seed_adds_rank(self):
        # When not distributed, rank=0
        assert distributed_seed(42) == 42

    def test_seed_everything_deterministic(self):
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Cleanup is safe to call when not initialised
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_noop_when_not_initialised(self):
        cleanup_distributed()  # should not raise

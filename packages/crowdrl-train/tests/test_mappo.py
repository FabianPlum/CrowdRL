"""Tests for MAPPO PPO update logic.

NOTE: Tests that call updater.update() run as subprocesses to avoid
PyTorch MKL access violations inside the pytest process on Windows.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from crowdrl_train.config import PPOConfig
from crowdrl_train.mappo import MAPPOUpdater
from crowdrl_train.networks import ActorCritic


def _run_python(code: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run Python code in a subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


class TestMAPPOUpdater:
    def test_update_produces_finite_metrics(self):
        """A single update should produce finite loss values."""
        result = _run_python("""
            import numpy as np, torch
            from crowdrl_train.buffer import FlatBatch
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.networks import ActorCritic

            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            updater = MAPPOUpdater(ac, PPOConfig(n_epochs=2), torch.device("cpu"))
            batch = FlatBatch(
                obs=torch.randn(50, 79), actions_raw=torch.randn(50, 4) * 0.5,
                log_probs=torch.randn(50) * 0.5, advantages=torch.randn(50),
                returns=torch.randn(50), values=torch.randn(50),
            )
            metrics = updater.update(batch)
            assert np.isfinite(metrics["policy_loss"])
            assert np.isfinite(metrics["value_loss"])
            assert np.isfinite(metrics["entropy"])
            assert np.isfinite(metrics["approx_kl"])
            assert np.isfinite(metrics["explained_variance"])
            print("PASS")
        """)
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "PASS" in result.stdout

    def test_kl_early_stopping(self):
        """Very low target_kl should trigger early stopping."""
        result = _run_python("""
            import torch
            from crowdrl_train.buffer import FlatBatch
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.networks import ActorCritic

            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            updater = MAPPOUpdater(
                ac, PPOConfig(n_epochs=100, target_kl=1e-10), torch.device("cpu")
            )
            batch = FlatBatch(
                obs=torch.randn(50, 79), actions_raw=torch.randn(50, 4) * 0.5,
                log_probs=torch.randn(50) * 0.5, advantages=torch.randn(50),
                returns=torch.randn(50), values=torch.randn(50),
            )
            metrics = updater.update(batch)
            assert metrics["n_epochs_actual"] < 100
            print("PASS")
        """)
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "PASS" in result.stdout

    def test_no_value_clip_by_default(self):
        """Default config should use MSE, not clipped value loss."""
        cfg = PPOConfig()
        assert cfg.use_value_clip is False

    def test_lr_decay(self, tiny_actor_critic: ActorCritic):
        """Linear LR schedule should decay learning rate."""
        updater = MAPPOUpdater(
            tiny_actor_critic, PPOConfig(lr_actor=1e-3, lr_critic=1e-3), torch.device("cpu")
        )
        initial_lr = updater.actor_optimizer.param_groups[0]["lr"]
        updater.update_learning_rate(0.5)
        mid_lr = updater.actor_optimizer.param_groups[0]["lr"]
        updater.update_learning_rate(1.0)
        final_lr = updater.actor_optimizer.param_groups[0]["lr"]

        assert mid_lr == pytest.approx(initial_lr * 0.5)
        assert final_lr == pytest.approx(0.0)

    def test_separate_actor_critic_optimizers(self, tiny_actor_critic: ActorCritic):
        """Actor and critic should have separate optimizers."""
        updater = MAPPOUpdater(
            tiny_actor_critic,
            PPOConfig(lr_actor=1e-3, lr_critic=5e-3),
            torch.device("cpu"),
        )
        assert updater.actor_optimizer.param_groups[0]["lr"] == 1e-3
        assert updater.critic_optimizer.param_groups[0]["lr"] == 5e-3

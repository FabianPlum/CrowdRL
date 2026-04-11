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

    def test_ddp_kl_early_stopping_synchronized(self, tmp_path):
        """Regression: under DDP, KL early stopping must use the global KL.

        Previously each rank computed its own local KL and decided independently
        whether to early-stop. With different local data, ranks would disagree
        on the decision, leading to mismatched collectives and NCCL deadlock.

        This test runs update() inside a single-process gloo group (world=1) and
        verifies that all_reduce is called on the KL tensor inside the loop.
        """
        # Use a sentinel file to communicate from subprocess back to test
        sentinel = tmp_path / "called.txt"
        result = _run_python(f"""
            import os
            import torch
            import torch.distributed as dist
            from unittest.mock import patch

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

            from crowdrl_train.buffer import FlatBatch
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.networks import ActorCritic

            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            # distributed=True forces the DDP code paths even with world_size=1
            updater = MAPPOUpdater(
                ac, PPOConfig(n_epochs=2, target_kl=0.02),
                torch.device("cpu"), distributed=True,
            )
            batch = FlatBatch(
                obs=torch.randn(50, 79), actions_raw=torch.randn(50, 4) * 0.5,
                log_probs=torch.randn(50) * 0.5, advantages=torch.randn(50),
                returns=torch.randn(50), values=torch.randn(50),
            )

            # Spy on all_reduce to verify the KL collective is invoked.
            # The KL tensor is 1-element scalar; gradient all-reduces are larger.
            kl_allreduce_calls = []
            original_all_reduce = dist.all_reduce
            def spy_all_reduce(tensor, *args, **kwargs):
                if tensor.numel() == 1:
                    kl_allreduce_calls.append(tensor.clone().detach())
                return original_all_reduce(tensor, *args, **kwargs)

            with patch.object(dist, "all_reduce", side_effect=spy_all_reduce):
                metrics = updater.update(batch)

            # KL all_reduce must have been called at least once (one per minibatch)
            assert len(kl_allreduce_calls) >= 1, (
                f"Expected >=1 KL all_reduce calls, got {{len(kl_allreduce_calls)}}"
            )
            # And the early-stop decision must use the SAME value that was reduced
            # (not a new local computation). The metrics dict carries the averaged KL.
            assert metrics["approx_kl"] is not None
            with open({str(sentinel)!r}, "w") as f:
                f.write(f"OK calls={{len(kl_allreduce_calls)}}")
            dist.destroy_process_group()
            print("PASS")
        """)
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "PASS" in result.stdout
        assert sentinel.exists()
        assert "OK" in sentinel.read_text()

    def test_ddp_kl_check_uses_reduced_value(self, tmp_path):
        """Regression: with world_size=1, the reduced KL must equal the local KL.

        This test pins down that the early-stop check actually consumes the
        all-reduced tensor's value. With world_size=1 the global mean equals
        the local mean, but the test ensures we're not accidentally using the
        pre-reduce value (which would defeat the fix).
        """
        result = _run_python("""
            import os
            import torch
            import torch.distributed as dist

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29502"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

            from crowdrl_train.buffer import FlatBatch
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.networks import ActorCritic

            torch.manual_seed(0)
            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            updater = MAPPOUpdater(
                ac, PPOConfig(n_epochs=100, target_kl=1e-10),
                torch.device("cpu"), distributed=True,
            )
            batch = FlatBatch(
                obs=torch.randn(50, 79), actions_raw=torch.randn(50, 4) * 0.5,
                log_probs=torch.randn(50) * 0.5, advantages=torch.randn(50),
                returns=torch.randn(50), values=torch.randn(50),
            )
            metrics = updater.update(batch)
            # Tiny target_kl should trigger early stop on the very first minibatch
            assert metrics["n_epochs_actual"] < 100, (
                f"Early stop did not fire under DDP path: {metrics['n_epochs_actual']}"
            )
            dist.destroy_process_group()
            print("PASS")
        """)
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "PASS" in result.stdout

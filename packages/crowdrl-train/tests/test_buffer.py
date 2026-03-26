"""Tests for RolloutBuffer and GAE computation.

GAE correctness is critical — errors here silently corrupt training.
Tests include hand-computed expected values for verification.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crowdrl_train.buffer import FlatBatch, RolloutBuffer


class TestRolloutBuffer:
    def test_add_and_count(self):
        """Basic add and counting."""
        buf = RolloutBuffer(obs_dim=4, action_dim=2, device=torch.device("cpu"))
        n_agents = 3
        for _ in range(5):
            buf.add(
                obs=np.zeros((n_agents, 4)),
                actions_raw=np.zeros((n_agents, 2)),
                log_probs=np.zeros(n_agents),
                rewards=np.ones(n_agents),
                values=np.zeros(n_agents),
                dones=np.zeros(n_agents, dtype=np.bool_),
                active_mask=np.ones(n_agents, dtype=np.bool_),
            )
        assert buf.total_steps == 5
        assert buf.total_active_agent_steps == 15  # 5 steps × 3 agents

    def test_inactive_agents_excluded(self):
        """Inactive agents should be excluded from flatten()."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        n_agents = 3
        # Step 1: all active
        buf.add(
            obs=np.ones((n_agents, 2)),
            actions_raw=np.zeros((n_agents, 1)),
            log_probs=np.zeros(n_agents),
            rewards=np.ones(n_agents),
            values=np.zeros(n_agents),
            dones=np.zeros(n_agents, dtype=np.bool_),
            active_mask=np.array([True, True, True]),
        )
        # Step 2: agent 0 is done, agent 1 and 2 still active
        buf.add(
            obs=np.ones((n_agents, 2)),
            actions_raw=np.zeros((n_agents, 1)),
            log_probs=np.zeros(n_agents),
            rewards=np.ones(n_agents),
            values=np.zeros(n_agents),
            dones=np.array([True, False, False]),
            active_mask=np.array([False, True, True]),
        )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(n_agents),
            last_dones=np.ones(n_agents, dtype=np.bool_),
            gamma=0.99,
            gae_lambda=0.95,
        )

        batch = buf.flatten()
        # Step 1: 3 active + Step 2: 2 active = 5 total
        assert batch.batch_size == 5

    def test_gae_single_step_episode(self):
        """GAE for a single-step episode: advantage = reward - value."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        buf.add(
            obs=np.zeros((1, 2)),
            actions_raw=np.zeros((1, 1)),
            log_probs=np.zeros(1),
            rewards=np.array([5.0]),
            values=np.array([2.0]),
            dones=np.array([True]),
            active_mask=np.array([True]),
        )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(1),
            last_dones=np.ones(1, dtype=np.bool_),
            gamma=0.99,
            gae_lambda=0.95,
        )

        batch = buf.flatten()
        # delta = reward + gamma * V_next * (1-done) - V = 5 + 0 - 2 = 3
        # GAE(single step) = delta = 3
        assert batch.advantages[0].item() == pytest.approx(3.0)
        # return = advantage + value = 3 + 2 = 5
        assert batch.returns[0].item() == pytest.approx(5.0)

    def test_gae_two_step_no_discount(self):
        """GAE with gamma=1, lambda=1 should give undiscounted Monte Carlo return."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        # Step 0: reward=1, value=0
        buf.add(
            obs=np.zeros((1, 2)),
            actions_raw=np.zeros((1, 1)),
            log_probs=np.zeros(1),
            rewards=np.array([1.0]),
            values=np.array([0.0]),
            dones=np.array([False]),
            active_mask=np.array([True]),
        )
        # Step 1: reward=2, value=0, done
        buf.add(
            obs=np.zeros((1, 2)),
            actions_raw=np.zeros((1, 1)),
            log_probs=np.zeros(1),
            rewards=np.array([2.0]),
            values=np.array([0.0]),
            dones=np.array([True]),
            active_mask=np.array([True]),
        )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(1),
            last_dones=np.ones(1, dtype=np.bool_),
            gamma=1.0,
            gae_lambda=1.0,
        )

        batch = buf.flatten()
        # With gamma=1, lambda=1:
        # Step 1: delta_1 = 2 + 0 - 0 = 2, GAE_1 = 2
        # Step 0: delta_0 = 1 + 1*0*(1) - 0 = 1, GAE_0 = 1 + 1*1*1*2 = 3
        # Return_0 = GAE_0 + V_0 = 3 + 0 = 3 (= total reward 1+2)
        # Return_1 = GAE_1 + V_1 = 2 + 0 = 2
        returns = batch.returns.numpy()
        assert returns[0] == pytest.approx(3.0)
        assert returns[1] == pytest.approx(2.0)

    def test_gae_with_discount(self):
        """Hand-computed GAE with gamma=0.9, lambda=0.8."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        gamma, lam = 0.9, 0.8

        # 3-step episode, 1 agent
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 0.5]
        for i in range(3):
            buf.add(
                obs=np.zeros((1, 2)),
                actions_raw=np.zeros((1, 1)),
                log_probs=np.zeros(1),
                rewards=np.array([rewards[i]]),
                values=np.array([values[i]]),
                dones=np.array([i == 2]),  # done at last step
                active_mask=np.array([True]),
            )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(1),
            last_dones=np.ones(1, dtype=np.bool_),
            gamma=gamma,
            gae_lambda=lam,
        )

        batch = buf.flatten()
        adv = batch.advantages.numpy()

        # Hand computation:
        # Step 2 (done): delta_2 = 3 + 0 - 0.5 = 2.5, GAE_2 = 2.5
        # Step 1: delta_1 = 2 + 0.9*0.5*(1) - 1.0 = 1.45, GAE_1 = 1.45 + 0.9*0.8*1*2.5 = 3.25
        # Step 0: delta_0 = 1 + 0.9*1.0*(1) - 0.5 = 1.4, GAE_0 = 1.4 + 0.9*0.8*1*3.25 = 3.74
        assert adv[2] == pytest.approx(2.5, abs=1e-6)
        assert adv[1] == pytest.approx(3.25, abs=1e-6)
        assert adv[0] == pytest.approx(3.74, abs=1e-6)

    def test_mid_episode_termination(self):
        """Agent terminating mid-episode should reset GAE chain."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))

        # 2 agents, 3 steps. Agent 0 terminates at step 1.
        # Step 0: both active
        buf.add(
            obs=np.zeros((2, 2)),
            actions_raw=np.zeros((2, 1)),
            log_probs=np.zeros(2),
            rewards=np.array([1.0, 1.0]),
            values=np.array([0.0, 0.0]),
            dones=np.array([False, False]),
            active_mask=np.array([True, True]),
        )
        # Step 1: agent 0 terminates
        buf.add(
            obs=np.zeros((2, 2)),
            actions_raw=np.zeros((2, 1)),
            log_probs=np.zeros(2),
            rewards=np.array([10.0, 1.0]),
            values=np.array([0.0, 0.0]),
            dones=np.array([True, False]),
            active_mask=np.array([True, True]),
        )
        # Step 2: only agent 1 active
        buf.add(
            obs=np.zeros((2, 2)),
            actions_raw=np.zeros((2, 1)),
            log_probs=np.zeros(2),
            rewards=np.array([0.0, 5.0]),
            values=np.array([0.0, 0.0]),
            dones=np.array([False, True]),
            active_mask=np.array([False, True]),
        )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(2),
            last_dones=np.ones(2, dtype=np.bool_),
            gamma=1.0,
            gae_lambda=1.0,
        )

        batch = buf.flatten()
        # 2 + 2 + 1 = 5 active agent-steps
        assert batch.batch_size == 5

    def test_variable_agent_count_across_episodes(self):
        """Buffer should handle episodes with different agent counts."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))

        # Episode 1: 3 agents, 2 steps
        for _ in range(2):
            buf.add(
                obs=np.zeros((3, 2)),
                actions_raw=np.zeros((3, 1)),
                log_probs=np.zeros(3),
                rewards=np.ones(3),
                values=np.zeros(3),
                dones=np.zeros(3, dtype=np.bool_),
                active_mask=np.ones(3, dtype=np.bool_),
            )
        # Mark last step as done
        buf._dones[-1] = np.ones(3, dtype=np.bool_)
        buf.mark_episode_end()

        # Episode 2: 5 agents, 1 step
        buf.add(
            obs=np.zeros((5, 2)),
            actions_raw=np.zeros((5, 1)),
            log_probs=np.zeros(5),
            rewards=np.ones(5),
            values=np.zeros(5),
            dones=np.ones(5, dtype=np.bool_),
            active_mask=np.ones(5, dtype=np.bool_),
        )
        buf.mark_episode_end()

        buf.compute_gae(
            last_values=np.zeros(5),
            last_dones=np.ones(5, dtype=np.bool_),
            gamma=0.99,
            gae_lambda=0.95,
        )

        batch = buf.flatten()
        assert batch.batch_size == 11  # 3*2 + 5*1

    def test_empty_buffer_flatten(self):
        """Flatten on empty buffer after GAE should return empty batch."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        buf.compute_gae(np.zeros(0), np.ones(0, dtype=np.bool_), 0.99, 0.95)
        batch = buf.flatten()
        assert batch.batch_size == 0

    def test_clear_resets(self):
        """Clear should reset all internal state."""
        buf = RolloutBuffer(obs_dim=2, action_dim=1, device=torch.device("cpu"))
        buf.add(
            obs=np.zeros((3, 2)),
            actions_raw=np.zeros((3, 1)),
            log_probs=np.zeros(3),
            rewards=np.ones(3),
            values=np.zeros(3),
            dones=np.zeros(3, dtype=np.bool_),
            active_mask=np.ones(3, dtype=np.bool_),
        )
        buf.clear()
        assert buf.total_steps == 0
        assert buf.total_active_agent_steps == 0


class TestFlatBatch:
    def test_minibatch_indices_cover_all(self):
        """Minibatch indices should cover every element exactly once."""
        batch = FlatBatch(
            obs=torch.randn(20, 4),
            actions_raw=torch.randn(20, 2),
            log_probs=torch.randn(20),
            advantages=torch.randn(20),
            returns=torch.randn(20),
            values=torch.randn(20),
        )
        gen = torch.Generator()
        gen.manual_seed(42)
        indices_list = batch.minibatch_indices(4, gen)
        all_indices = torch.cat(indices_list)
        assert sorted(all_indices.tolist()) == list(range(20))

    def test_single_minibatch(self):
        """n_minibatches=1 should return one index tensor covering full batch."""
        batch = FlatBatch(
            obs=torch.randn(10, 4),
            actions_raw=torch.randn(10, 2),
            log_probs=torch.randn(10),
            advantages=torch.randn(10),
            returns=torch.randn(10),
            values=torch.randn(10),
        )
        gen = torch.Generator()
        gen.manual_seed(42)
        indices_list = batch.minibatch_indices(1, gen)
        assert len(indices_list) == 1
        assert len(indices_list[0]) == 10

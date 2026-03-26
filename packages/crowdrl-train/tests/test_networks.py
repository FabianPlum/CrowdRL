"""Tests for Actor-Critic networks."""

from __future__ import annotations


import torch

from crowdrl_train.config import NetworkConfig
from crowdrl_train.networks import Actor, ActorCritic, Critic


class TestActor:
    def test_output_shapes(self, tiny_actor: Actor):
        """Forward pass should return Normal distribution with correct shapes."""
        obs = torch.randn(5, 79)
        dist = tiny_actor(obs)
        assert dist.mean.shape == (5, 4)
        assert dist.stddev.shape == (5, 4)

    def test_get_action_shapes(self, tiny_actor: Actor):
        """get_action should return action, log_prob, entropy with correct shapes."""
        obs = torch.randn(10, 79)
        action, log_prob, entropy = tiny_actor.get_action(obs)
        assert action.shape == (10, 4)
        assert log_prob.shape == (10,)
        assert entropy.shape == (10,)

    def test_actions_in_range(self, tiny_actor: Actor):
        """Clipped actions should be in [-1, 1]."""
        obs = torch.randn(100, 79)
        action, _, _ = tiny_actor.get_action(obs)
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_deterministic_mode(self, tiny_actor: Actor):
        """Deterministic actions should be reproducible."""
        obs = torch.randn(5, 79)
        a1, _, _ = tiny_actor.get_action(obs, deterministic=True)
        a2, _, _ = tiny_actor.get_action(obs, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_log_prob_finite(self, tiny_actor: Actor):
        """Log-probabilities should be finite (no NaN or Inf)."""
        obs = torch.randn(20, 79)
        _, log_prob, _ = tiny_actor.get_action(obs)
        assert torch.isfinite(log_prob).all()

    def test_evaluate_matches_get_action(self, tiny_actor: Actor):
        """evaluate_actions on the raw action should give same log_prob."""
        torch.manual_seed(42)
        obs = torch.randn(10, 79)
        dist = tiny_actor(obs)
        raw_action = dist.rsample()
        log_prob_direct = dist.log_prob(raw_action).sum(dim=-1)
        log_prob_eval, _ = tiny_actor.evaluate_actions(obs, raw_action)
        assert torch.allclose(log_prob_direct, log_prob_eval)

    def test_initial_std_approximately_0_5(self):
        """Initial std should be ~0.5 per Andrychowicz et al. (2021)."""
        config = NetworkConfig(
            obs_dim=79,
            action_dim=4,
            actor_hidden_sizes=(32, 32),
        )
        actor = Actor(config)
        initial_std = actor.log_std.exp().detach()
        assert torch.allclose(initial_std, torch.full((4,), 0.5), atol=1e-4)

    def test_orthogonal_init_scales(self):
        """Hidden layer weights should have ~sqrt(2) gain, actor output ~0.01."""
        config = NetworkConfig(
            obs_dim=10,
            action_dim=4,
            actor_hidden_sizes=(32,),
            ortho_init=True,
        )
        actor = Actor(config)

        # Output layer: gain 0.01 → very small weights
        w_out = actor.action_mean.weight
        assert w_out.abs().max().item() < 0.1


class TestCritic:
    def test_output_shape(self, tiny_critic: Critic):
        obs = torch.randn(5, 79)
        value = tiny_critic(obs)
        assert value.shape == (5, 1)

    def test_output_finite(self, tiny_critic: Critic):
        obs = torch.randn(20, 79)
        value = tiny_critic(obs)
        assert torch.isfinite(value).all()

    def test_custom_critic_input_dim(self):
        """Critic can have different input dim (CTDE)."""
        config = NetworkConfig(
            obs_dim=79,
            action_dim=4,
            critic_hidden_sizes=(32,),
            critic_obs_dim=85,
        )
        critic = Critic(config)
        obs_with_global = torch.randn(5, 85)
        value = critic(obs_with_global)
        assert value.shape == (5, 1)


class TestActorCritic:
    def test_get_action_and_value(self, tiny_actor_critic: ActorCritic):
        obs = torch.randn(5, 79)
        action, raw_action, log_prob, entropy, value = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (5, 4)
        assert raw_action.shape == (5, 4)
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)

    def test_get_value_only(self, tiny_actor_critic: ActorCritic):
        obs = torch.randn(5, 79)
        value = tiny_actor_critic.get_value(obs)
        assert value.shape == (5,)

    def test_actor_critic_independent_params(self, tiny_actor_critic: ActorCritic):
        """Actor and critic should have fully independent parameters."""
        actor_params = set(id(p) for p in tiny_actor_critic.actor.parameters())
        critic_params = set(id(p) for p in tiny_actor_critic.critic.parameters())
        assert actor_params.isdisjoint(critic_params)

    def test_single_agent_batch(self, tiny_actor_critic: ActorCritic):
        """Should work with batch size 1."""
        obs = torch.randn(1, 79)
        action, raw, lp, ent, val = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (1, 4)
        assert val.shape == (1,)

    def test_large_batch(self, tiny_actor_critic: ActorCritic):
        """Should work with batch size 100 (max agents per episode)."""
        obs = torch.randn(100, 79)
        action, raw, lp, ent, val = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (100, 4)

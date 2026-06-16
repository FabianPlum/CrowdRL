"""Tests for Actor-Critic networks."""

from __future__ import annotations

import math

import torch
from torch.distributions import Normal

from crowdrl_train.config import NetworkConfig
from crowdrl_train.networks import (
    _LOG_STD_MAX,
    Actor,
    ActorCritic,
    Critic,
    _squashed_log_prob,
)


class TestActor:
    def test_output_shapes(self, tiny_actor: Actor):
        """Forward pass should return Normal distribution with correct shapes."""
        obs = torch.randn(5, 80)
        dist = tiny_actor(obs)
        assert dist.mean.shape == (5, 4)
        assert dist.stddev.shape == (5, 4)

    def test_get_action_shapes(self, tiny_actor: Actor):
        """get_action should return action, log_prob, entropy with correct shapes."""
        obs = torch.randn(10, 80)
        action, log_prob, entropy = tiny_actor.get_action(obs)
        assert action.shape == (10, 4)
        assert log_prob.shape == (10,)
        assert entropy.shape == (10,)

    def test_actions_in_range(self, tiny_actor: Actor):
        """tanh-squashed actions should be strictly within (-1, 1)."""
        obs = torch.randn(100, 80)
        action, _, _ = tiny_actor.get_action(obs)
        assert action.abs().max() < 1.0

    def test_deterministic_mode(self, tiny_actor: Actor):
        """Deterministic actions should be reproducible."""
        obs = torch.randn(5, 80)
        a1, _, _ = tiny_actor.get_action(obs, deterministic=True)
        a2, _, _ = tiny_actor.get_action(obs, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_log_prob_finite(self, tiny_actor: Actor):
        """Log-probabilities should be finite (no NaN or Inf)."""
        obs = torch.randn(20, 80)
        _, log_prob, _ = tiny_actor.get_action(obs)
        assert torch.isfinite(log_prob).all()

    def test_evaluate_matches_squashed_log_prob(self, tiny_actor: Actor):
        """evaluate_actions reproduces the tanh-squashed log-prob of the raw sample."""
        torch.manual_seed(42)
        obs = torch.randn(10, 80)
        dist = tiny_actor(obs)
        raw_action = dist.rsample()
        ref = _squashed_log_prob(dist, raw_action)
        log_prob_eval, _ = tiny_actor.evaluate_actions(obs, raw_action)
        assert torch.allclose(ref, log_prob_eval)

    def test_squashed_log_prob_matches_naive_formula(self):
        """Stable Jacobian == naive log(1 - tanh^2) form in the numerically safe regime."""
        torch.manual_seed(1)
        mean = torch.randn(7, 4)
        std = torch.full((7, 4), 0.5)
        dist = Normal(mean, std)
        u = mean + std * torch.randn(7, 4)  # moderate |u|, no underflow
        stable = _squashed_log_prob(dist, u)
        naive = (dist.log_prob(u) - torch.log(1.0 - torch.tanh(u) ** 2 + 1e-12)).sum(dim=-1)
        assert torch.allclose(stable, naive, atol=1e-4)

    def test_squashed_log_prob_finite_at_extreme_u(self):
        """Jacobian correction stays finite for large |u| (the std-runaway regime)."""
        dist = Normal(torch.zeros(1, 4), torch.ones(1, 4))
        u = torch.tensor([[-30.0, -12.0, 12.0, 30.0]])
        lp = _squashed_log_prob(dist, u)
        assert torch.isfinite(lp).all()

    def test_deterministic_is_tanh_mean(self, tiny_actor: Actor):
        """Deterministic action equals tanh(action mean) -- matches the ONNX export."""
        obs = torch.randn(6, 80)
        action, _, _ = tiny_actor.get_action(obs, deterministic=True)
        dist = tiny_actor(obs)
        assert torch.allclose(action, torch.tanh(dist.mean), atol=1e-6)

    def test_current_std_post_clamp(self):
        """current_std reflects the post-clamp exploration scale (the logged quantity)."""
        config = NetworkConfig(obs_dim=80, action_dim=4, actor_hidden_sizes=(32, 32))
        actor = Actor(config)
        with torch.no_grad():
            actor.log_std.fill_(5.0)  # above _LOG_STD_MAX
        std = actor.current_std()
        assert torch.allclose(std, torch.full((4,), math.exp(_LOG_STD_MAX)), atol=1e-6)

    def test_initial_std_approximately_0_5(self):
        """Initial std should be ~0.5 per Andrychowicz et al. (2021)."""
        config = NetworkConfig(
            obs_dim=80,
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
        obs = torch.randn(5, 80)
        value = tiny_critic(obs)
        assert value.shape == (5, 1)

    def test_output_finite(self, tiny_critic: Critic):
        obs = torch.randn(20, 80)
        value = tiny_critic(obs)
        assert torch.isfinite(value).all()

    def test_custom_critic_input_dim(self):
        """Critic can have different input dim (CTDE)."""
        config = NetworkConfig(
            obs_dim=80,
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
        obs = torch.randn(5, 80)
        action, raw_action, log_prob, entropy, value = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (5, 4)
        assert raw_action.shape == (5, 4)
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)

    def test_get_value_only(self, tiny_actor_critic: ActorCritic):
        obs = torch.randn(5, 80)
        value = tiny_actor_critic.get_value(obs)
        assert value.shape == (5,)

    def test_collection_logprob_matches_reeval(self, tiny_actor_critic: ActorCritic):
        """PPO consistency: stored log_prob == evaluate_actions(stored raw action).

        At collection time the importance ratio must be exactly 1, so the
        log-prob returned by get_action_and_value has to equal the one recomputed
        by evaluate_actions on the stored pre-squash action.
        """
        torch.manual_seed(0)
        obs = torch.randn(8, 80)
        _, raw, log_prob, _, _ = tiny_actor_critic.get_action_and_value(obs)
        re_lp, _ = tiny_actor_critic.actor.evaluate_actions(obs, raw)
        assert torch.allclose(log_prob, re_lp, atol=1e-6)

    def test_actor_critic_independent_params(self, tiny_actor_critic: ActorCritic):
        """Actor and critic should have fully independent parameters."""
        actor_params = set(id(p) for p in tiny_actor_critic.actor.parameters())
        critic_params = set(id(p) for p in tiny_actor_critic.critic.parameters())
        assert actor_params.isdisjoint(critic_params)

    def test_single_agent_batch(self, tiny_actor_critic: ActorCritic):
        """Should work with batch size 1."""
        obs = torch.randn(1, 80)
        action, raw, lp, ent, val = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (1, 4)
        assert val.shape == (1,)

    def test_large_batch(self, tiny_actor_critic: ActorCritic):
        """Should work with batch size 100 (max agents per episode)."""
        obs = torch.randn(100, 80)
        action, raw, lp, ent, val = tiny_actor_critic.get_action_and_value(obs)
        assert action.shape == (100, 4)

"""Shared test fixtures for crowdrl-train tests."""

from __future__ import annotations

import pytest

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
from crowdrl_env.spawner import SpawnConfig

from crowdrl_train.config import NetworkConfig, PPOConfig, TrainConfig
from crowdrl_train.networks import Actor, ActorCritic, Critic


@pytest.fixture
def tiny_network_config() -> NetworkConfig:
    """Minimal network config for fast tests."""
    return NetworkConfig(
        obs_dim=79,
        action_dim=4,
        actor_hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
    )


@pytest.fixture
def tiny_actor(tiny_network_config: NetworkConfig) -> Actor:
    return Actor(tiny_network_config)


@pytest.fixture
def tiny_critic(tiny_network_config: NetworkConfig) -> Critic:
    return Critic(tiny_network_config)


@pytest.fixture
def tiny_actor_critic(tiny_network_config: NetworkConfig) -> ActorCritic:
    return ActorCritic(tiny_network_config)


@pytest.fixture
def tiny_env_config() -> CrowdEnvConfig:
    """Minimal env config: Tier 0, 3-5 agents, 20 steps."""
    return CrowdEnvConfig(
        geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
        spawn=SpawnConfig(n_agents_range=(3, 5)),
        max_steps=20,
    )


@pytest.fixture
def tiny_env(tiny_env_config: CrowdEnvConfig) -> CrowdEnv:
    return CrowdEnv(config=tiny_env_config, seed=42)


@pytest.fixture
def tiny_train_config(tiny_env_config: CrowdEnvConfig) -> TrainConfig:
    """Minimal training config for fast integration tests."""
    return TrainConfig(
        network=NetworkConfig(
            obs_dim=79,
            action_dim=4,
            actor_hidden_sizes=(32, 32),
            critic_hidden_sizes=(32, 32),
        ),
        ppo=PPOConfig(
            n_epochs=2,
            n_minibatches=1,
        ),
        env=tiny_env_config,
        total_timesteps=500,
        seed=42,
        normalize_obs=True,
        normalize_rewards=False,
        checkpoint_interval=100,
    )

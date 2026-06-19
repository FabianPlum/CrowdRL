"""crowdrl-env: Gymnasium training environment for CrowdRL."""

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.eval_metrics import aggregate_metrics, compute_episode_metrics
from crowdrl_env.geometry_generator import (
    GeneratedGeometry,
    GeometryConfig,
    GeometryTier,
    generate_geometry,
)
from crowdrl_env.reward import RewardConfig, RewardState, compute_rewards
from crowdrl_env.solvability import SolvabilityMode, verify_solvability
from crowdrl_env.spawner import SpawnConfig, SpawnResult, spawn_agents

__all__ = [
    "CrowdEnv",
    "CrowdEnvConfig",
    "GeneratedGeometry",
    "GeometryConfig",
    "GeometryTier",
    "RewardConfig",
    "RewardState",
    "SolvabilityMode",
    "SpawnConfig",
    "SpawnResult",
    "aggregate_metrics",
    "compute_episode_metrics",
    "compute_rewards",
    "generate_geometry",
    "spawn_agents",
    "verify_solvability",
]

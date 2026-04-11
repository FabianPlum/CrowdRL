"""crowdrl-train: MAPPO training infrastructure for CrowdRL.

Implements Multi-Agent PPO with parameter sharing for training
pedestrian navigation policies in procedurally generated environments.
"""

from crowdrl_train.config import (
    CurriculumConfig,
    CurriculumPhase,
    DDPConfig,
    LogConfig,
    NetworkConfig,
    PPOConfig,
    TrainConfig,
    VecEnvConfig,
)
from crowdrl_train.curriculum import CurriculumManager, EpisodeStats
from crowdrl_train.export import export_onnx, verify_onnx
from crowdrl_train.networks import Actor, ActorCritic, Critic
from crowdrl_train.rollout_collector import RolloutCollector
from crowdrl_train.train import train
from crowdrl_train.vec_env import SubprocVecEnv

__all__ = [
    "Actor",
    "ActorCritic",
    "Critic",
    "CurriculumConfig",
    "CurriculumManager",
    "CurriculumPhase",
    "DDPConfig",
    "EpisodeStats",
    "LogConfig",
    "NetworkConfig",
    "PPOConfig",
    "RolloutCollector",
    "SubprocVecEnv",
    "TrainConfig",
    "VecEnvConfig",
    "export_onnx",
    "train",
    "verify_onnx",
]

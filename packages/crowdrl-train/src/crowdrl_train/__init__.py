"""crowdrl-train: MAPPO training infrastructure for CrowdRL.

Implements Multi-Agent PPO with parameter sharing for training
pedestrian navigation policies in procedurally generated environments.
"""

from crowdrl_train.config import (
    CurriculumConfig,
    CurriculumPhase,
    LogConfig,
    NetworkConfig,
    PPOConfig,
    TrainConfig,
)
from crowdrl_train.curriculum import CurriculumManager, EpisodeStats
from crowdrl_train.export import export_onnx, verify_onnx
from crowdrl_train.networks import Actor, ActorCritic, Critic
from crowdrl_train.train import train

__all__ = [
    "Actor",
    "ActorCritic",
    "Critic",
    "CurriculumConfig",
    "CurriculumManager",
    "CurriculumPhase",
    "EpisodeStats",
    "LogConfig",
    "NetworkConfig",
    "PPOConfig",
    "TrainConfig",
    "export_onnx",
    "train",
    "verify_onnx",
]

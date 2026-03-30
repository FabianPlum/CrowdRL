"""PyTorch-accelerated GPU environments for CrowdRL training."""

import os
import sys

# Workaround for Windows MAX_PATH (260 char) limitation.
# TorchInductor/Triton generates temp files with very long names that exceed
# the limit under the default %TEMP% path. Use a short cache path instead.
if sys.platform == "win32" and "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\tmp\\torchinductor"

from crowdrl_torch.batched_env import BatchedTorchEnv
from crowdrl_torch.episode_factory import make_episode_factory
from crowdrl_torch.normalizer import TorchRunningNormalizer
from crowdrl_torch.torch_collector import TorchRolloutCollector
from crowdrl_torch.types import EnvConfig, TorchWorldState

__all__ = [
    "BatchedTorchEnv",
    "EnvConfig",
    "TorchRolloutCollector",
    "TorchRunningNormalizer",
    "TorchWorldState",
    "make_episode_factory",
]

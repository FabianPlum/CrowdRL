"""Training metrics logging.

TensorBoard backend by default (no account needed).
Optional W&B support via extra dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol


class TrainLogger(Protocol):
    """Logging interface for training metrics."""

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_scalars(self, tag_values: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class TensorBoardLogger:
    """TensorBoard logging via torch.utils.tensorboard."""

    def __init__(self, log_dir: str, run_name: str | None = None):
        from torch.utils.tensorboard import SummaryWriter

        if run_name is None:
            run_name = f"crowdrl_{int(time.time())}"
        self._writer = SummaryWriter(log_dir=str(Path(log_dir) / run_name))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_values: dict[str, float], step: int) -> None:
        for tag, value in tag_values.items():
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()


class ConsoleLogger:
    """Simple console-only logger for quick experiments."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        print(f"[{step}] {tag}: {value:.4f}")

    def log_scalars(self, tag_values: dict[str, float], step: int) -> None:
        parts = [f"{tag}: {value:.4f}" for tag, value in tag_values.items()]
        print(f"[{step}] {' | '.join(parts)}")

    def close(self) -> None:
        pass


def create_logger(backend: str = "tensorboard", **kwargs) -> TrainLogger:
    """Factory function for creating a logger.

    Parameters
    ----------
    backend : 'tensorboard', 'console', or 'wandb'
    **kwargs : passed to the logger constructor
    """
    if backend == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif backend == "console":
        return ConsoleLogger()
    elif backend == "wandb":
        raise NotImplementedError("W&B logging not yet implemented. Use tensorboard.")
    else:
        raise ValueError(f"Unknown logging backend: {backend}")

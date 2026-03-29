"""Batched PyTorch environment with async CPU resets.

Manages N_ENVS environments on GPU. Each step processes all environments
using batched tensor operations. Episode resets are handled asynchronously
by a CPU thread pool running Shapely geometry generation.

Replaces ``SubprocVecEnv`` entirely — no subprocess pipes, no IPC overhead.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from crowdrl_torch.geometry_repr import prepare_reset_data
from crowdrl_torch.observation import build_observations
from crowdrl_torch.step import batched_step
from crowdrl_torch.types import EnvConfig, TorchWorldState


class BatchedTorchEnv:
    """GPU-accelerated batched environment with async CPU resets.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    config : EnvConfig
        Static environment configuration.
    make_episode_fn : callable
        ``(seed: int) -> dict`` — CPU-side function that generates an episode.
    device : torch.device or str
        Device for all tensors (e.g. "cuda" or "cpu").
    seed : int
        Base random seed.
    n_reset_workers : int
        Number of CPU threads for async reset generation.
    compile_step : bool
        If True, apply ``torch.compile(mode="reduce-overhead")`` to the
        step function for kernel fusion and CUDA graph support.
    """

    def __init__(
        self,
        n_envs: int,
        config: EnvConfig,
        make_episode_fn: Any,
        device: torch.device | str = "cpu",
        seed: int = 42,
        n_reset_workers: int = 8,
        compile_step: bool = False,
    ):
        self.n_envs = n_envs
        self.config = config
        self.make_episode_fn = make_episode_fn
        self.device = torch.device(device)
        self.seed = seed
        self._seed_counter = seed

        self._step_fn = batched_step
        self._compiled = False
        if compile_step:
            self._step_fn = torch.compile(batched_step, mode="reduce-overhead")
            self._compiled = True

        self.reset_pool = ThreadPoolExecutor(max_workers=n_reset_workers)
        self._pending_resets: dict[int, Future] = {}

        self.states: TorchWorldState | None = None
        self.episode_over: torch.Tensor | None = None

    def reset_all(self) -> tuple[TorchWorldState, torch.Tensor]:
        """Reset all environments synchronously. Call once at training start.

        Returns
        -------
        states : TorchWorldState — batched (E, N, ...)
        observations : (E, N, obs_dim)
        """
        all_data = []
        for _ in range(self.n_envs):
            data = self._generate_reset_data(self._next_seed())
            all_data.append(data)

        self.states = self._stack_reset_data(all_data)

        # No episodes are over right after reset
        self.episode_over = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

        # Build initial observations
        obs = build_observations(
            self.states.positions,
            self.states.velocities,
            self.states.torso_orientations,
            self.states.head_orientations,
            self.states.shoulder_widths,
            self.states.chest_depths,
            self.states.goal_positions,
            self.states.active_mask,
            self.states.n_agents,
            self.states.wall_segments,
            self.states.n_segments,
            self.config,
        )

        return self.states, obs

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TorchWorldState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments.

        Parameters
        ----------
        actions : (E, N, 4)

        Returns
        -------
        states, observations, rewards, terminated, truncated

        Notes
        -----
        ``self.episode_over`` is a (E,) bool tensor that is True only for envs
        whose episode **newly** completed this step (transitioned from having
        active agents to having none). Envs that are idle waiting for an async
        reset are NOT flagged, preventing ghost episode reports.
        """
        # Remember which envs had active agents BEFORE this step
        had_active = self.states.active_mask.any(dim=-1)  # (E,)

        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
        self.states, obs, rewards, terminated, truncated = self._step_fn(
            self.states, actions, self.config
        )
        if self._compiled:
            self.states = self.states.clone()

        # Episode over = transitioned from active to inactive THIS step.
        # Envs already idle (waiting for async reset) are NOT flagged.
        any_active = self.states.active_mask.any(dim=-1)  # (E,)
        self.episode_over = had_active & ~any_active

        # Initiate async resets for finished envs
        for env_idx in self.episode_over.nonzero(as_tuple=False).flatten().tolist():
            if env_idx not in self._pending_resets:
                seed = self._next_seed()
                self._pending_resets[env_idx] = self.reset_pool.submit(
                    self._generate_reset_data, seed
                )

        # Apply completed resets
        self._apply_completed_resets()

        return self.states, obs, rewards, terminated, truncated

    def get_episode_over_mask(self) -> torch.Tensor:
        """Return (E,) bool mask of envs with no active agents."""
        return ~self.states.active_mask.any(dim=-1)

    def warmup(self, n_steps: int = 3) -> bool:
        """Trigger torch.compile tracing with a few dummy steps.

        Call after ``reset_all()`` to absorb the one-time compilation cost.
        If compilation fails (e.g. missing Triton/CUDA headers), falls back
        to the eager step function and prints a warning.

        Returns True if compiled, False if fell back to eager.
        """
        import warnings

        if not self._compiled:
            return False

        actions = torch.zeros(
            self.n_envs,
            self.config.max_agents,
            4,
            device=self.device,
            dtype=torch.float32,
        )
        try:
            for _ in range(n_steps):
                torch.compiler.cudagraph_mark_step_begin()
                self.states, _, _, _, _ = self._step_fn(self.states, actions, self.config)
                self.states = self.states.clone()
            return True
        except Exception as e:
            warnings.warn(
                f"torch.compile warmup failed ({type(e).__name__}: {e}). "
                f"Falling back to eager execution.",
                stacklevel=2,
            )
            self._step_fn = batched_step
            self._compiled = False
            return False

    def close(self) -> None:
        """Shut down the reset thread pool."""
        self.reset_pool.shutdown(wait=False)

    # --- Internal ---

    def _next_seed(self) -> int:
        self._seed_counter += 1
        return self._seed_counter

    def _generate_reset_data(self, seed: int) -> dict[str, NDArray[np.float32]]:
        """Generate episode on CPU (runs in thread pool)."""
        raw = self.make_episode_fn(seed)
        return prepare_reset_data(
            positions=raw["positions"],
            velocities=raw["velocities"],
            torso_orientations=raw["torso_orientations"],
            head_orientations=raw["head_orientations"],
            shoulder_widths=raw["shoulder_widths"],
            chest_depths=raw["chest_depths"],
            goal_positions=raw["goal_positions"],
            preferred_speeds=raw["preferred_speeds"],
            wall_segments=raw["wall_segments"],
            max_agents=self.config.max_agents,
            max_segments=self.config.max_segments,
        )

    def _data_to_tensors(self, data: dict[str, NDArray]) -> dict[str, torch.Tensor]:
        """Convert padded numpy arrays to tensors on device."""
        dev = self.device
        return {
            "positions": torch.tensor(data["positions"], dtype=torch.float32, device=dev),
            "velocities": torch.tensor(data["velocities"], dtype=torch.float32, device=dev),
            "torso_orientations": torch.tensor(
                data["torso_orientations"], dtype=torch.float32, device=dev
            ),
            "head_orientations": torch.tensor(
                data["head_orientations"], dtype=torch.float32, device=dev
            ),
            "shoulder_widths": torch.tensor(
                data["shoulder_widths"], dtype=torch.float32, device=dev
            ),
            "chest_depths": torch.tensor(data["chest_depths"], dtype=torch.float32, device=dev),
            "goal_positions": torch.tensor(
                data["goal_positions"], dtype=torch.float32, device=dev
            ),
            "preferred_speeds": torch.tensor(
                data["preferred_speeds"], dtype=torch.float32, device=dev
            ),
            "active_mask": torch.tensor(data["active_mask"], dtype=torch.bool, device=dev),
            "wall_segments": torch.tensor(data["wall_segments"], dtype=torch.float32, device=dev),
            "n_segments": torch.tensor(data["n_segments"], dtype=torch.int32, device=dev),
            "n_agents": torch.tensor(data["n_agents"], dtype=torch.int32, device=dev),
            "goal_distances": torch.tensor(
                data["goal_distances"], dtype=torch.float32, device=dev
            ),
        }

    def _stack_reset_data(self, all_data: list[dict]) -> TorchWorldState:
        """Stack list of reset data dicts into a batched TorchWorldState."""
        all_tensors = [self._data_to_tensors(d) for d in all_data]
        max_agents = self.config.max_agents
        dev = self.device

        def stack_field(key: str) -> torch.Tensor:
            return torch.stack([t[key] for t in all_tensors])

        return TorchWorldState(
            positions=stack_field("positions"),
            velocities=stack_field("velocities"),
            torso_orientations=stack_field("torso_orientations"),
            head_orientations=stack_field("head_orientations"),
            shoulder_widths=stack_field("shoulder_widths"),
            chest_depths=stack_field("chest_depths"),
            goal_positions=stack_field("goal_positions"),
            preferred_speeds=stack_field("preferred_speeds"),
            active_mask=stack_field("active_mask"),
            cumulative_terminated=torch.zeros(
                (self.n_envs, max_agents), dtype=torch.bool, device=dev
            ),
            wall_segments=stack_field("wall_segments"),
            n_segments=stack_field("n_segments"),
            prev_velocities=torch.zeros(
                (self.n_envs, max_agents, 2), dtype=torch.float32, device=dev
            ),
            prev_goal_distances=stack_field("goal_distances"),
            prev_accelerations=torch.zeros(
                (self.n_envs, max_agents, 2), dtype=torch.float32, device=dev
            ),
            prev_headings=stack_field("torso_orientations"),
            prev_heading_changes=torch.zeros(
                (self.n_envs, max_agents), dtype=torch.float32, device=dev
            ),
            prev_actions=torch.zeros(
                (self.n_envs, max_agents, 4), dtype=torch.float32, device=dev
            ),
            n_agents=stack_field("n_agents"),
            step_count=torch.zeros(self.n_envs, dtype=torch.int32, device=dev),
        )

    def _apply_completed_resets(self) -> None:
        """Apply any completed async resets to the batched state."""
        completed = []
        for env_idx, future in self._pending_resets.items():
            if future.done():
                data = future.result()
                tensors = self._data_to_tensors(data)

                # Direct slice assignment — simpler than JAX's tree.map
                self.states.positions[env_idx] = tensors["positions"]
                self.states.velocities[env_idx] = tensors["velocities"]
                self.states.torso_orientations[env_idx] = tensors["torso_orientations"]
                self.states.head_orientations[env_idx] = tensors["head_orientations"]
                self.states.shoulder_widths[env_idx] = tensors["shoulder_widths"]
                self.states.chest_depths[env_idx] = tensors["chest_depths"]
                self.states.goal_positions[env_idx] = tensors["goal_positions"]
                self.states.preferred_speeds[env_idx] = tensors["preferred_speeds"]
                self.states.active_mask[env_idx] = tensors["active_mask"]
                self.states.cumulative_terminated[env_idx] = False
                self.states.wall_segments[env_idx] = tensors["wall_segments"]
                self.states.n_segments[env_idx] = tensors["n_segments"]
                self.states.prev_velocities[env_idx] = 0.0
                self.states.prev_goal_distances[env_idx] = tensors["goal_distances"]
                self.states.prev_accelerations[env_idx] = 0.0
                self.states.prev_headings[env_idx] = tensors["torso_orientations"]
                self.states.prev_heading_changes[env_idx] = 0.0
                self.states.prev_actions[env_idx] = 0.0
                self.states.n_agents[env_idx] = tensors["n_agents"]
                self.states.step_count[env_idx] = 0

                completed.append(env_idx)

        for idx in completed:
            del self._pending_resets[idx]

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

        # Per-env geometry tier name (e.g. "TIER_3B"), updated on every reset.
        # Exposed to collectors for per-tier episode statistics.
        self.env_tiers: list[str] = ["unknown"] * n_envs

    def reset_all(self) -> tuple[TorchWorldState, torch.Tensor]:
        """Reset all environments synchronously. Call once at training start.

        Returns
        -------
        states : TorchWorldState — batched (E, N, ...)
        observations : (E, N, obs_dim)
        """
        all_data = []
        for env_idx in range(self.n_envs):
            data, tier_name = self._generate_reset_data(self._next_seed())
            all_data.append(data)
            self.env_tiers[env_idx] = tier_name

        self.states = self._stack_reset_data(all_data)

        # When neighbor memory is on, seed the persistent neighbor-ID table
        # with a first-step match so the initial observation sees populated
        # slots rather than all -1. Without this seed, slots would remain
        # unpopulated until after the first step, giving the policy a stale
        # all-empty neighbor context on reset (fine for commit 2, but a
        # clean seed is simpler to reason about for commits 4/5).
        if self.config.use_neighbor_memory:
            from crowdrl_torch.sensing import match_persistent_neighbors

            self.states.neighbor_ids = match_persistent_neighbors(
                self.states.positions,
                self.states.neighbor_ids,
                self.states.active_mask,
                self.states.n_agents,
                sensing_radius=self.config.neighbor_sensing_radius,
                config=self.config,
            )

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
            waypoints=self.states.waypoints,
            n_waypoints=self.states.n_waypoints,
            waypoint_cursor=self.states.waypoint_cursor,
            waypoint_path_lengths=self.states.waypoint_path_lengths,
            spawn_positions=self.states.spawn_positions,
            initial_goal_distances=self.states.initial_goal_distances,
            cumulative_path_length=self.states.cumulative_path_length,
            pos_history=self.states.pos_history,
            gdist_history=self.states.gdist_history,
            preferred_speeds=self.states.preferred_speeds,
            step_count=self.states.step_count,
            neighbor_ids=self.states.neighbor_ids,
            neighbor_vel_history=self.states.neighbor_vel_history,
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

        # Apply completed resets and rebuild observations for reset envs
        # so callers see the new episode's initial obs instead of stale zeros.
        reset_envs = self._apply_completed_resets()
        if reset_envs and self.config.use_neighbor_memory:
            # Re-seed the persistent neighbor slots for envs that just reset.
            # We just cleared them to -1 in _apply_completed_resets; running
            # the matcher on the full batch is idempotent for mid-episode
            # envs (their positions haven't moved since batched_step already
            # matched them) and populates the cleared reset envs.
            from crowdrl_torch.sensing import match_persistent_neighbors

            self.states.neighbor_ids = match_persistent_neighbors(
                self.states.positions,
                self.states.neighbor_ids,
                self.states.active_mask,
                self.states.n_agents,
                sensing_radius=self.config.neighbor_sensing_radius,
                config=self.config,
            )
        if reset_envs:
            r = torch.tensor(reset_envs, device=self.device)
            fresh_obs = build_observations(
                self.states.positions[r],
                self.states.velocities[r],
                self.states.torso_orientations[r],
                self.states.head_orientations[r],
                self.states.shoulder_widths[r],
                self.states.chest_depths[r],
                self.states.goal_positions[r],
                self.states.active_mask[r],
                self.states.n_agents[r],
                self.states.wall_segments[r],
                self.states.n_segments[r],
                self.config,
                waypoints=self.states.waypoints[r],
                n_waypoints=self.states.n_waypoints[r],
                waypoint_cursor=self.states.waypoint_cursor[r],
                waypoint_path_lengths=self.states.waypoint_path_lengths[r],
                spawn_positions=self.states.spawn_positions[r],
                initial_goal_distances=self.states.initial_goal_distances[r],
                cumulative_path_length=self.states.cumulative_path_length[r],
                pos_history=self.states.pos_history[r],
                gdist_history=self.states.gdist_history[r],
                preferred_speeds=self.states.preferred_speeds[r],
                step_count=self.states.step_count[r],
                neighbor_ids=self.states.neighbor_ids[r],
                neighbor_vel_history=self.states.neighbor_vel_history[r],
            )
            obs[r] = fresh_obs

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

    def _generate_reset_data(self, seed: int) -> tuple[dict[str, NDArray[np.float32]], str]:
        """Generate episode on CPU (runs in thread pool).

        Returns ``(prepared_data, tier_name)`` so per-env tier tracking can
        survive the trip through the thread pool.
        """
        raw = self.make_episode_fn(seed)
        tier_name = str(raw.get("tier", "unknown"))
        data = prepare_reset_data(
            positions=raw["positions"],
            velocities=raw["velocities"],
            torso_orientations=raw["torso_orientations"],
            head_orientations=raw["head_orientations"],
            shoulder_widths=raw["shoulder_widths"],
            chest_depths=raw["chest_depths"],
            masses=raw.get("masses"),
            goal_positions=raw["goal_positions"],
            preferred_speeds=raw["preferred_speeds"],
            wall_segments=raw["wall_segments"],
            max_agents=self.config.max_agents,
            max_segments=self.config.max_segments,
            waypoints=raw.get("waypoints"),
            n_waypoints=raw.get("n_waypoints"),
            waypoint_path_lengths=raw.get("waypoint_path_lengths"),
            max_waypoints=self.config.max_waypoints,
            memory_window=self.config.temporal_memory_window,
        )
        return data, tier_name

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
            "masses": torch.tensor(data["masses"], dtype=torch.float32, device=dev),
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
            "waypoints": torch.tensor(data["waypoints"], dtype=torch.float32, device=dev),
            "n_waypoints": torch.tensor(data["n_waypoints"], dtype=torch.int32, device=dev),
            "waypoint_path_lengths": torch.tensor(
                data["waypoint_path_lengths"], dtype=torch.float32, device=dev
            ),
            "spawn_positions": torch.tensor(
                data["spawn_positions"], dtype=torch.float32, device=dev
            ),
            "initial_goal_distances": torch.tensor(
                data["initial_goal_distances"], dtype=torch.float32, device=dev
            ),
            "pos_history": torch.tensor(data["pos_history"], dtype=torch.float32, device=dev),
            "gdist_history": torch.tensor(data["gdist_history"], dtype=torch.float32, device=dev),
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
            masses=stack_field("masses"),
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
            waypoints=stack_field("waypoints"),
            n_waypoints=stack_field("n_waypoints"),
            waypoint_cursor=torch.zeros((self.n_envs, max_agents), dtype=torch.int32, device=dev),
            waypoint_path_lengths=stack_field("waypoint_path_lengths"),
            n_agents=stack_field("n_agents"),
            step_count=torch.zeros(self.n_envs, dtype=torch.int32, device=dev),
            stuck_window_step=torch.zeros(
                (self.n_envs, max_agents), dtype=torch.int32, device=dev
            ),
            stuck_window_start_dist=stack_field("goal_distances"),
            spawn_positions=stack_field("spawn_positions"),
            initial_goal_distances=stack_field("initial_goal_distances"),
            cumulative_path_length=torch.zeros(
                (self.n_envs, max_agents), dtype=torch.float32, device=dev
            ),
            pos_history=stack_field("pos_history"),
            gdist_history=stack_field("gdist_history"),
            neighbor_ids=torch.full(
                (self.n_envs, max_agents, self.config.k_neighbours),
                -1,
                dtype=torch.int32,
                device=dev,
            ),
            neighbor_vel_history=torch.zeros(
                (
                    self.n_envs,
                    max_agents,
                    self.config.neighbor_vel_history_window + 1,
                    self.config.k_neighbours,
                    2,
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )

    def _apply_completed_resets(self) -> list[int]:
        """Apply any completed async resets to the batched state.

        Returns list of env indices that were reset.
        """
        completed = []
        for env_idx, future in self._pending_resets.items():
            if future.done():
                data, tier_name = future.result()
                self.env_tiers[env_idx] = tier_name
                tensors = self._data_to_tensors(data)

                # Direct slice assignment — simpler than JAX's tree.map
                self.states.positions[env_idx] = tensors["positions"]
                self.states.velocities[env_idx] = tensors["velocities"]
                self.states.torso_orientations[env_idx] = tensors["torso_orientations"]
                self.states.head_orientations[env_idx] = tensors["head_orientations"]
                self.states.shoulder_widths[env_idx] = tensors["shoulder_widths"]
                self.states.chest_depths[env_idx] = tensors["chest_depths"]
                self.states.masses[env_idx] = tensors["masses"]
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
                self.states.waypoints[env_idx] = tensors["waypoints"]
                self.states.n_waypoints[env_idx] = tensors["n_waypoints"]
                self.states.waypoint_cursor[env_idx] = 0
                self.states.waypoint_path_lengths[env_idx] = tensors["waypoint_path_lengths"]
                self.states.n_agents[env_idx] = tensors["n_agents"]
                self.states.step_count[env_idx] = 0
                self.states.stuck_window_step[env_idx] = 0
                self.states.stuck_window_start_dist[env_idx] = tensors["goal_distances"]
                self.states.spawn_positions[env_idx] = tensors["spawn_positions"]
                self.states.initial_goal_distances[env_idx] = tensors["initial_goal_distances"]
                self.states.cumulative_path_length[env_idx] = 0.0
                self.states.pos_history[env_idx] = tensors["pos_history"]
                self.states.gdist_history[env_idx] = tensors["gdist_history"]
                self.states.neighbor_ids[env_idx] = -1
                self.states.neighbor_vel_history[env_idx] = 0.0

                completed.append(env_idx)

        for idx in completed:
            del self._pending_resets[idx]

        return completed

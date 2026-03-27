"""Vectorized CrowdEnv using subprocess workers.

Runs N CrowdEnv instances in separate processes to bypass the GIL.
The main process performs central GPU inference — workers only run
env.reset() and env.step(). Communication uses multiprocessing.Pipe
for low-latency point-to-point messaging.

Design constraints:
- Windows uses ``spawn`` — every worker arg must be picklable.
  CrowdEnvConfig (frozen dataclass of primitives/enums) pickles fine.
  CrowdEnv is created *inside* each worker, never sent across processes.
- Variable n_agents per env: observations are sent as numpy arrays
  with shapes that differ across workers.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection

from numpy.typing import NDArray

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig


# ---------------------------------------------------------------------------
# Message types (command, payload) for main ↔ worker communication
# ---------------------------------------------------------------------------
CMD_RESET = "reset"
CMD_STEP = "step"
CMD_RECONFIGURE = "reconfigure"
CMD_CLOSE = "close"


@dataclass
class StepResult:
    """Result of a single env.step() call."""

    obs: NDArray
    rewards: NDArray
    terminated: NDArray
    truncated: NDArray
    info: dict


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_fn(
    pipe: Connection,
    initial_config: CrowdEnvConfig,
    seed: int,
) -> None:
    """Worker loop: owns one CrowdEnv, responds to commands from main."""
    env = CrowdEnv(config=initial_config, seed=seed)

    try:
        while True:
            cmd, payload = pipe.recv()

            if cmd == CMD_RESET:
                obs, info = env.reset()
                pipe.send((obs, info))

            elif cmd == CMD_STEP:
                actions = payload
                obs, rewards, terminated, truncated, info = env.step(actions)
                pipe.send((obs, rewards, terminated, truncated, info))

            elif cmd == CMD_RECONFIGURE:
                new_config, new_seed = payload
                env = CrowdEnv(config=new_config, seed=new_seed)
                obs, info = env.reset()
                pipe.send((obs, info))

            elif cmd == CMD_CLOSE:
                pipe.send(None)
                break

            else:
                pipe.send(("error", f"Unknown command: {cmd}"))

    except Exception:
        try:
            pipe.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# Main-process vectorized env
# ---------------------------------------------------------------------------


class SubprocVecEnv:
    """Vectorized CrowdEnv backed by subprocess workers.

    Each worker runs its own CrowdEnv instance. The main process
    sends commands (reset / step / reconfigure / close) and receives
    results via per-worker Pipes.

    Parameters
    ----------
    env_configs : per-worker configs (or a single config broadcast to all)
    seeds : per-worker random seeds
    """

    def __init__(
        self,
        env_configs: list[CrowdEnvConfig] | CrowdEnvConfig,
        seeds: list[int],
    ):
        if isinstance(env_configs, CrowdEnvConfig):
            env_configs = [env_configs] * len(seeds)
        if len(env_configs) != len(seeds):
            raise ValueError(f"Got {len(env_configs)} configs but {len(seeds)} seeds")

        self._n_envs = len(seeds)
        self._main_pipes: list[Connection] = []
        self._workers: list[mp.Process] = []
        self._closed = False

        ctx = mp.get_context("spawn")
        for i, (cfg, seed) in enumerate(zip(env_configs, seeds)):
            main_conn, worker_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_fn,
                args=(worker_conn, cfg, seed),
                name=f"CrowdEnv-worker-{i}",
                daemon=True,
            )
            p.start()
            worker_conn.close()  # main only uses main_conn
            self._main_pipes.append(main_conn)
            self._workers.append(p)

    # -- Properties ----------------------------------------------------------

    @property
    def n_envs(self) -> int:
        return self._n_envs

    # -- Commands ------------------------------------------------------------

    def reset_all(self) -> list[tuple[NDArray, dict]]:
        """Reset every env. Returns [(obs, info), ...] per env."""
        for pipe in self._main_pipes:
            pipe.send((CMD_RESET, None))
        results = [self._recv(i) for i in range(self._n_envs)]
        return results  # each is (obs, info)

    def step(self, actions_list: list[NDArray]) -> list[StepResult]:
        """Step all envs in parallel.

        Parameters
        ----------
        actions_list : list of (n_agents_i, action_dim) arrays, one per env.

        Returns
        -------
        list of StepResult, one per env.
        """
        assert len(actions_list) == self._n_envs
        for pipe, actions in zip(self._main_pipes, actions_list):
            pipe.send((CMD_STEP, actions))

        results = []
        for i in range(self._n_envs):
            obs, rewards, terminated, truncated, info = self._recv(i)
            results.append(StepResult(obs, rewards, terminated, truncated, info))
        return results

    def reset_env(self, env_idx: int, config: CrowdEnvConfig, seed: int) -> tuple[NDArray, dict]:
        """Reconfigure and reset a single worker."""
        self._main_pipes[env_idx].send((CMD_RECONFIGURE, (config, seed)))
        return self._recv(env_idx)

    def update_all_configs(
        self, config: CrowdEnvConfig, base_seed: int
    ) -> list[tuple[NDArray, dict]]:
        """Reconfigure all workers with a new config (e.g. curriculum change).

        Each worker gets ``seed = base_seed + env_idx``.
        Returns [(obs, info), ...] from the fresh resets.
        """
        for i, pipe in enumerate(self._main_pipes):
            pipe.send((CMD_RECONFIGURE, (config, base_seed + i)))
        return [self._recv(i) for i in range(self._n_envs)]

    def close(self) -> None:
        """Terminate all workers."""
        if self._closed:
            return
        self._closed = True
        for pipe in self._main_pipes:
            try:
                pipe.send((CMD_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for pipe in self._main_pipes:
            try:
                pipe.recv()
            except (BrokenPipeError, OSError, EOFError):
                pass
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for pipe in self._main_pipes:
            pipe.close()

    def __del__(self) -> None:
        self.close()

    # -- Helpers -------------------------------------------------------------

    def _recv(self, env_idx: int):
        """Receive a result from worker, raising on errors."""
        result = self._main_pipes[env_idx].recv()
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], str)
            and result[0] == "error"
        ):
            raise RuntimeError(f"Worker {env_idx} raised an error:\n{result[1]}")
        return result

"""Main MAPPO training loop for CrowdRL.

Orchestrates: env creation → rollout collection → PPO update →
curriculum advancement → logging → checkpointing.

This implements Step 3 of the CrowdRL build path (see project plan §3.6):
Train initial policies on Tier 0–2 environments using MAPPO with parameter
sharing. Deliverable: trained .onnx policy files and training logs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from crowdrl_env.crowd_env import CrowdEnv

from crowdrl_train.buffer import RolloutBuffer
from crowdrl_train.config import TrainConfig
from crowdrl_train.curriculum import CurriculumManager, EpisodeStats
from crowdrl_train.export import export_onnx
from crowdrl_train.logger import create_logger
from crowdrl_train.mappo import MAPPOUpdater
from crowdrl_train.networks import ActorCritic
from crowdrl_train.normalizer import RewardNormalizer, RunningNormalizer


def save_checkpoint(
    path: Path,
    actor_critic: ActorCritic,
    updater: MAPPOUpdater,
    obs_normalizer: RunningNormalizer | None,
    reward_normalizer: RewardNormalizer | None,
    curriculum: CurriculumManager,
    total_steps: int,
    rollout_count: int,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "actor_critic": actor_critic.state_dict(),
        "actor_optimizer": updater.actor_optimizer.state_dict(),
        "critic_optimizer": updater.critic_optimizer.state_dict(),
        "total_steps": total_steps,
        "rollout_count": rollout_count,
        "curriculum": curriculum.state_dict(),
    }
    if obs_normalizer is not None:
        checkpoint["obs_normalizer"] = obs_normalizer.state_dict()
    if reward_normalizer is not None:
        checkpoint["reward_normalizer"] = reward_normalizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    actor_critic: ActorCritic,
    updater: MAPPOUpdater,
    obs_normalizer: RunningNormalizer | None,
    reward_normalizer: RewardNormalizer | None,
    curriculum: CurriculumManager,
) -> tuple[int, int]:
    """Load a training checkpoint. Returns (total_steps, rollout_count)."""
    checkpoint = torch.load(path, weights_only=False)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    updater.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    updater.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    curriculum.load_state_dict(checkpoint["curriculum"])
    if obs_normalizer is not None and "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
    if reward_normalizer is not None and "reward_normalizer" in checkpoint:
        reward_normalizer.load_state_dict(checkpoint["reward_normalizer"])
    return checkpoint["total_steps"], checkpoint["rollout_count"]


def collect_episode(
    env: CrowdEnv,
    actor_critic: ActorCritic,
    buffer: RolloutBuffer,
    obs_normalizer: RunningNormalizer | None,
    reward_normalizer: RewardNormalizer | None,
    device: torch.device,
) -> dict:
    """Collect a single complete episode into the buffer.

    Returns episode statistics dict.
    """
    obs, info = env.reset()
    n_agents = info["n_agents"]

    episode_rewards = np.zeros(n_agents, dtype=np.float64)
    cumulative_terminated = np.zeros(n_agents, dtype=np.bool_)
    active_mask = np.ones(n_agents, dtype=np.bool_)
    episode_length = 0
    episode_over = False

    while not episode_over:
        # Normalise observations
        if obs_normalizer is not None:
            obs_active = obs[active_mask]
            if obs_active.shape[0] > 0:
                obs_normalizer.update(obs_active)
            obs_norm = obs_normalizer.normalize(obs)
        else:
            obs_norm = obs

        # Forward pass (no grad for collection)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
            actions, actions_raw, log_probs, _entropy, values = actor_critic.get_action_and_value(
                obs_t
            )

        actions_np = actions.cpu().numpy()
        actions_raw_np = actions_raw.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        # Step environment
        next_obs, rewards, terminated, truncated, step_info = env.step(actions_np)
        dones = terminated | truncated
        episode_length += 1

        # Normalise rewards
        if reward_normalizer is not None:
            rewards = reward_normalizer.normalize(rewards, dones)

        # Store transition
        buffer.add(
            obs=obs_norm,
            actions_raw=actions_raw_np,
            log_probs=log_probs_np,
            rewards=rewards,
            values=values_np,
            dones=dones,
            active_mask=active_mask.copy(),
        )

        episode_rewards += rewards * active_mask
        cumulative_terminated |= terminated
        active_mask = ~(cumulative_terminated | truncated)

        obs = next_obs
        episode_over = step_info.get("episode_over", False)

    buffer.mark_episode_end()

    # Episode statistics
    n_reached_goal = int(cumulative_terminated.sum())
    goal_rate = n_reached_goal / n_agents if n_agents > 0 else 0.0

    return {
        "n_agents": n_agents,
        "episode_length": episode_length,
        "goal_rate": goal_rate,
        "n_reached_goal": n_reached_goal,
        "mean_reward": float(episode_rewards.sum() / n_agents) if n_agents > 0 else 0.0,
        "total_reward": float(episode_rewards.sum()),
        "n_collisions": step_info.get("n_collisions", 0),
        "geometry_tier": info.get("geometry_tier", "unknown"),
    }


def train(config: TrainConfig, resume_from: str | Path | None = None) -> Path:
    """Main training loop.

    When ``config.vec_env.n_envs > 1``, uses subprocess-parallel environments
    with central GPU inference for higher throughput. Falls back to the
    single-env sequential path when ``n_envs == 1``.

    Parameters
    ----------
    config : full training configuration
    resume_from : path to checkpoint to resume from (optional)

    Returns
    -------
    Path to the final exported ONNX policy
    """
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Initialise components ---
    actor_critic = ActorCritic(config.network).to(device)
    updater = MAPPOUpdater(actor_critic, config.ppo, device)
    buffer = RolloutBuffer(config.network.obs_dim, config.network.action_dim, device)
    curriculum = CurriculumManager(config.curriculum)

    obs_normalizer = (
        RunningNormalizer(shape=(config.network.obs_dim,)) if config.normalize_obs else None
    )
    reward_normalizer = (
        RewardNormalizer(gamma=config.ppo.gamma) if config.normalize_rewards else None
    )

    logger = create_logger(
        backend=config.log.backend,
        log_dir=config.log.log_dir,
    )

    # Resume from checkpoint
    total_steps = 0
    rollout_count = 0
    if resume_from is not None:
        total_steps, rollout_count = load_checkpoint(
            Path(resume_from),
            actor_critic,
            updater,
            obs_normalizer,
            reward_normalizer,
            curriculum,
        )

    # Save config for reproducibility
    config_path = Path(config.checkpoint_dir) / "config.json"
    config.save_json(config_path)

    n_envs = config.vec_env.n_envs

    print(f"Training MAPPO | device={device} | seed={config.seed} | n_envs={n_envs}")
    print(
        f"Network: actor {config.network.actor_hidden_sizes}, "
        f"critic {config.network.critic_hidden_sizes}"
    )
    print(f"Curriculum phase: {curriculum.current_phase.name}")
    print(f"Total timesteps target: {config.total_timesteps:,}")

    start_time = time.time()

    if n_envs > 1:
        _train_vec(
            config,
            device,
            actor_critic,
            updater,
            buffer,
            curriculum,
            obs_normalizer,
            reward_normalizer,
            logger,
            total_steps,
            rollout_count,
            start_time,
        )
    else:
        _train_single(
            config,
            device,
            actor_critic,
            updater,
            buffer,
            curriculum,
            obs_normalizer,
            reward_normalizer,
            logger,
            total_steps,
            rollout_count,
            start_time,
        )

    # --- Final checkpoint and export ---
    # (total_steps is local to the sub-functions; re-read from closure isn't
    #  possible, so final checkpoint is saved inside each sub-function)

    logger.close()
    onnx_path = Path(config.checkpoint_dir) / "policy.onnx"
    return onnx_path


def _train_single(
    config: TrainConfig,
    device: torch.device,
    actor_critic: ActorCritic,
    updater: MAPPOUpdater,
    buffer: RolloutBuffer,
    curriculum: CurriculumManager,
    obs_normalizer: RunningNormalizer | None,
    reward_normalizer: RewardNormalizer | None,
    logger,
    total_steps: int,
    rollout_count: int,
    start_time: float,
) -> None:
    """Single-env sequential training loop (original path)."""
    env_config = curriculum.make_env_config(config.env)
    env = CrowdEnv(config=env_config, seed=config.seed)

    while total_steps < config.total_timesteps:
        ep_stats = collect_episode(
            env, actor_critic, buffer, obs_normalizer, reward_normalizer, device
        )

        ep_agent_steps = buffer.total_active_agent_steps

        n_agents = ep_stats["n_agents"]
        last_values = np.zeros(n_agents, dtype=np.float64)
        last_dones = np.ones(n_agents, dtype=np.bool_)
        buffer.compute_gae(last_values, last_dones, config.ppo.gamma, config.ppo.gae_lambda)

        flat_batch = buffer.flatten()
        if flat_batch.batch_size > 0:
            update_metrics = updater.update(flat_batch)
        else:
            update_metrics = {}

        total_steps += ep_agent_steps
        rollout_count += 1
        buffer.clear()

        progress = total_steps / config.total_timesteps
        updater.update_learning_rate(progress)

        curriculum_stats = EpisodeStats(
            goal_rate=ep_stats["goal_rate"],
            n_agents=ep_stats["n_agents"],
            episode_length=ep_stats["episode_length"],
            mean_reward=ep_stats["mean_reward"],
        )
        phase_advanced = curriculum.report_episode(curriculum_stats)
        if phase_advanced:
            env_config = curriculum.make_env_config(config.env)
            env = CrowdEnv(config=env_config, seed=config.seed + rollout_count)
            print(f"\n>>> Curriculum advanced to: {curriculum.current_phase.name}")

        _log_and_checkpoint(
            config,
            logger,
            updater,
            actor_critic,
            obs_normalizer,
            reward_normalizer,
            curriculum,
            ep_stats,
            update_metrics,
            total_steps,
            rollout_count,
            start_time,
            progress,
        )

    _save_final(
        config,
        actor_critic,
        updater,
        obs_normalizer,
        reward_normalizer,
        curriculum,
        total_steps,
        rollout_count,
    )


def _train_vec(
    config: TrainConfig,
    device: torch.device,
    actor_critic: ActorCritic,
    updater: MAPPOUpdater,
    buffer: RolloutBuffer,
    curriculum: CurriculumManager,
    obs_normalizer: RunningNormalizer | None,
    reward_normalizer: RewardNormalizer | None,
    logger,
    total_steps: int,
    rollout_count: int,
    start_time: float,
) -> None:
    """Multi-env parallel training loop with central GPU inference."""
    from crowdrl_train.rollout_collector import RolloutCollector
    from crowdrl_train.vec_env import SubprocVecEnv

    n_envs = config.vec_env.n_envs
    n_steps_per_collect = config.vec_env.n_steps_per_collect

    env_config = curriculum.make_env_config(config.env)
    seeds = [config.seed + i for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_config, seeds)

    collector = RolloutCollector(
        vec_env,
        actor_critic,
        obs_normalizer,
        reward_normalizer,
        device,
        obs_dim=config.network.obs_dim,
        action_dim=config.network.action_dim,
    )

    print(f"Vectorized training: {n_envs} workers, {n_steps_per_collect} steps/collect")

    try:
        while total_steps < config.total_timesteps:
            # Collect rollouts from all envs (per-env buffers)
            episode_stats_list = collector.collect(n_steps_per_collect)

            # GAE per env buffer + merge into single FlatBatch
            flat_batch = collector.compute_gae_and_flatten(
                config.ppo.gamma,
                config.ppo.gae_lambda,
            )

            # PPO update
            if flat_batch.batch_size > 0:
                update_metrics = updater.update(flat_batch)
            else:
                update_metrics = {}

            batch_steps = collector.total_active_agent_steps
            total_steps += batch_steps
            rollout_count += 1

            # LR schedule
            progress = total_steps / config.total_timesteps
            updater.update_learning_rate(progress)

            # Report all completed episodes to curriculum
            phase_advanced = False
            for ep_stats in episode_stats_list:
                cs = EpisodeStats(
                    goal_rate=ep_stats["goal_rate"],
                    n_agents=ep_stats["n_agents"],
                    episode_length=ep_stats["episode_length"],
                    mean_reward=ep_stats["mean_reward"],
                )
                if curriculum.report_episode(cs):
                    phase_advanced = True

            if phase_advanced:
                env_config = curriculum.make_env_config(config.env)
                new_seeds = [config.seed + rollout_count * n_envs + i for i in range(n_envs)]
                vec_env.update_all_configs(env_config, new_seeds[0])
                print(f"\n>>> Curriculum advanced to: {curriculum.current_phase.name}")

            # Logging — aggregate over completed episodes
            if episode_stats_list and rollout_count % config.log.log_interval == 0:
                agg = _aggregate_episode_stats(episode_stats_list)
                _log_and_checkpoint(
                    config,
                    logger,
                    updater,
                    actor_critic,
                    obs_normalizer,
                    reward_normalizer,
                    curriculum,
                    agg,
                    update_metrics,
                    total_steps,
                    rollout_count,
                    start_time,
                    progress,
                )
            elif rollout_count % config.log.log_interval == 0:
                _log_and_checkpoint(
                    config,
                    logger,
                    updater,
                    actor_critic,
                    obs_normalizer,
                    reward_normalizer,
                    curriculum,
                    {"goal_rate": 0, "mean_reward": 0, "episode_length": 0, "n_agents": 0},
                    update_metrics,
                    total_steps,
                    rollout_count,
                    start_time,
                    progress,
                )

    finally:
        vec_env.close()

    _save_final(
        config,
        actor_critic,
        updater,
        obs_normalizer,
        reward_normalizer,
        curriculum,
        total_steps,
        rollout_count,
    )


def _aggregate_episode_stats(stats_list: list[dict]) -> dict:
    """Compute mean stats over a list of episode stat dicts."""
    if not stats_list:
        return {"goal_rate": 0, "mean_reward": 0, "episode_length": 0, "n_agents": 0}
    return {
        "goal_rate": np.mean([s["goal_rate"] for s in stats_list]),
        "mean_reward": np.mean([s["mean_reward"] for s in stats_list]),
        "episode_length": np.mean([s["episode_length"] for s in stats_list]),
        "n_agents": np.mean([s["n_agents"] for s in stats_list]),
    }


def _log_and_checkpoint(
    config,
    logger,
    updater,
    actor_critic,
    obs_normalizer,
    reward_normalizer,
    curriculum,
    ep_stats,
    update_metrics,
    total_steps,
    rollout_count,
    start_time,
    progress,
) -> None:
    """Shared logging and checkpointing logic."""
    if rollout_count % config.log.log_interval == 0:
        elapsed = time.time() - start_time
        sps = total_steps / max(elapsed, 1e-8)

        metrics = {
            "episode/reward_mean": ep_stats["mean_reward"],
            "episode/goal_rate": ep_stats["goal_rate"],
            "episode/length": ep_stats["episode_length"],
            "episode/n_agents": ep_stats["n_agents"],
            "train/total_steps": total_steps,
            "train/steps_per_second": sps,
            "train/progress": progress,
            "curriculum/phase": curriculum.current_phase_idx,
            "curriculum/rolling_goal_rate": curriculum.rolling_goal_rate,
        }
        metrics.update({f"ppo/{k}": v for k, v in update_metrics.items()})

        if updater.actor_optimizer.param_groups:
            metrics["train/lr_actor"] = updater.actor_optimizer.param_groups[0]["lr"]

        logger.log_scalars(metrics, total_steps)

    if rollout_count % config.checkpoint_interval == 0:
        ckpt_path = Path(config.checkpoint_dir) / f"checkpoint_{rollout_count}.pt"
        save_checkpoint(
            ckpt_path,
            actor_critic,
            updater,
            obs_normalizer,
            reward_normalizer,
            curriculum,
            total_steps,
            rollout_count,
        )

    if rollout_count % 10 == 0:
        elapsed = time.time() - start_time
        print(
            f"[{rollout_count:>5}] steps={total_steps:>10,} | "
            f"goal_rate={ep_stats['goal_rate']:.2f} | "
            f"reward={ep_stats['mean_reward']:>7.2f} | "
            f"agents={int(ep_stats['n_agents']):>3} | "
            f"phase={curriculum.current_phase.name} | "
            f"sps={total_steps / max(elapsed, 1):.0f}"
        )


def _save_final(
    config,
    actor_critic,
    updater,
    obs_normalizer,
    reward_normalizer,
    curriculum,
    total_steps,
    rollout_count,
) -> None:
    """Save final checkpoint and export ONNX."""
    final_ckpt = Path(config.checkpoint_dir) / "checkpoint_final.pt"
    save_checkpoint(
        final_ckpt,
        actor_critic,
        updater,
        obs_normalizer,
        reward_normalizer,
        curriculum,
        total_steps,
        rollout_count,
    )

    onnx_path = Path(config.checkpoint_dir) / "policy.onnx"
    export_onnx(actor_critic.actor, obs_normalizer, onnx_path)
    print(f"\nTraining complete. ONNX policy exported to: {onnx_path}")


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train MAPPO pedestrian navigation policies")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides defaults)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel env workers")
    args = parser.parse_args()

    # Load base config
    if args.config is not None:
        config = TrainConfig.load_json(args.config)
    else:
        config = TrainConfig()

    # Apply CLI overrides (reconstruct frozen dataclass)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    if overrides:
        d = config.to_dict()
        d.update(overrides)
        config = TrainConfig.from_dict(d)

    # Vec env override (separate since it's a nested config)
    if args.n_envs is not None:
        from crowdrl_train.config import VecEnvConfig

        config = TrainConfig.from_dict(
            {**config.to_dict(), "vec_env": VecEnvConfig(n_envs=args.n_envs)}
        )

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()

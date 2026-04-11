"""Training configuration dataclasses for CrowdRL MAPPO.

All hyperparameter defaults are grounded in the literature:
- Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative MARL"
- Andrychowicz et al. (2021) "What Matters In On-Policy RL?"
- Huang et al. (2022) "The 37 Implementation Details of PPO"

See plan/MAPPO_Literature_Review.md for full rationale and citations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryTier


@dataclass(frozen=True)
class NetworkConfig:
    """Actor-Critic network architecture.

    Separate actor and critic networks per Andrychowicz et al. (2021):
    separate outperformed shared trunk on 4/5 continuous control envs.
    """

    obs_dim: int = 79
    """Observation dimensionality (from ObsConfig.obs_dim)."""

    action_dim: int = 4
    """Action dimensionality (from ActionConfig.action_dim)."""

    actor_hidden_sizes: tuple[int, ...] = (256, 256)
    """Actor MLP hidden layer sizes."""

    critic_hidden_sizes: tuple[int, ...] = (256, 256)
    """Critic MLP hidden layer sizes. Can be wider than actor (Andrychowicz et al.)."""

    activation: str = "tanh"
    """Activation function. tanh outperforms ReLU for continuous control
    (Andrychowicz et al. 2021, 4/5 envs)."""

    ortho_init: bool = True
    """Orthogonal initialization with per-layer gain scaling (Huang et al. 2022,
    details #12-13). Gains: sqrt(2) hidden, 0.01 actor output, 1.0 critic output."""

    log_std_init: float = -0.6931471805599453
    """Initial log-std for Gaussian policy = log(0.5). Andrychowicz et al. (2021)
    found initial std ~0.5 optimal across most continuous control envs."""

    critic_obs_dim: int | None = None
    """Critic input dim if using CTDE (obs + global context). None = same as obs_dim.
    Yu et al. (2022): feature-pruned agent-specific global state is best."""


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters.

    Defaults follow MAPPO (Yu et al. 2022) unless noted otherwise.
    """

    lr_actor: float = 5e-4
    """Actor learning rate. Yu et al. (2022): 5e-4 to 7e-4."""

    lr_critic: float = 5e-4
    """Critic learning rate. Can be higher than actor (Yu et al. use up to 1e-3)."""

    adam_eps: float = 1e-5
    """Adam epsilon. MAPPO default; larger than PyTorch's 1e-8 for stability."""

    gamma: float = 0.99
    """Discount factor. Standard for episodic tasks (Yu et al., Andrychowicz et al.)."""

    gae_lambda: float = 0.95
    """GAE lambda. Yu et al. (2022): 0.95. Higher = lower bias, higher variance.
    Appropriate for our 500-step episodes."""

    clip_epsilon: float = 0.2
    """PPO clipping ratio. Yu et al. (2022): stay at or below 0.2 for MARL.
    Lower values (0.05-0.15) for harder multi-agent settings."""

    entropy_coef: float = 0.01
    """Entropy bonus coefficient. Low for continuous control; the learnable
    log_std already controls exploration (Huang et al. 2022)."""

    value_coef: float = 0.5
    """Value loss coefficient in combined loss."""

    max_grad_norm: float = 10.0
    """Global gradient norm clipping. Yu et al. (2022): 10.0.
    More permissive than CleanRL's 0.5; appropriate for multi-agent."""

    n_epochs: int = 10
    """PPO update epochs per rollout. Yu et al. (2022): 10-15 for easy tasks,
    5 for hard tasks. Excess reuse worsens MARL non-stationarity."""

    n_minibatches: int = 1
    """Number of minibatches per epoch. Yu et al. (2022): 'Avoid splitting data
    into mini-batches' for MARL. Full-batch update is default."""

    target_kl: float | None = 0.02
    """Early stopping on approx KL divergence. None = disabled."""

    lr_schedule: str = "cosine"
    """Learning rate schedule: 'cosine', 'linear' (decay to 0), or 'constant'."""

    normalize_advantages: bool = True
    """Per-batch advantage normalization (zero mean, unit std)."""

    use_value_clip: bool = False
    """PPO-style value loss clipping. Andrychowicz et al. (2021):
    'hurt performance regardless of threshold'. OFF by default."""

    use_huber_loss: bool = False
    """Huber loss for value function. Yu et al. use Huber(delta=10).
    MSE is the default per Andrychowicz et al."""

    huber_delta: float = 10.0
    """Huber loss delta if use_huber_loss=True."""


@dataclass(frozen=True)
class CurriculumPhase:
    """A single difficulty phase in the curriculum.

    Progression follows the Zone of Proximal Development principle
    (Narvekar et al. 2020, JMLR survey): moderate difficulty, not too easy or hard.
    """

    name: str
    """Human-readable phase name."""

    geometry_tiers: tuple[GeometryTier, ...]
    """Geometry tiers to sample from in this phase."""

    n_agents_range: tuple[int, int]
    """(min, max) agent count per episode."""

    goal_rate_threshold: float
    """Rolling goal rate must exceed this to advance. 0.0 = terminal phase."""

    tier_weights: tuple[float, ...] | None = None
    """Sampling weights for geometry_tiers. Must have the same length as
    geometry_tiers. None = uniform sampling. Use higher weights for harder
    tiers to bias toward more challenging episodes."""


DEFAULT_CURRICULUM_PHASES: tuple[CurriculumPhase, ...] = (
    CurriculumPhase(
        name="easy",
        geometry_tiers=(GeometryTier.TIER_0,),
        n_agents_range=(5, 15),
        goal_rate_threshold=0.7,
    ),
    CurriculumPhase(
        name="medium",
        geometry_tiers=(GeometryTier.TIER_0, GeometryTier.TIER_1),
        n_agents_range=(10, 30),
        goal_rate_threshold=0.6,
    ),
    CurriculumPhase(
        name="hard",
        geometry_tiers=(GeometryTier.TIER_1, GeometryTier.TIER_2),
        n_agents_range=(20, 50),
        goal_rate_threshold=0.5,
    ),
    CurriculumPhase(
        name="rooms",
        geometry_tiers=(GeometryTier.TIER_2, GeometryTier.TIER_3A),
        n_agents_range=(15, 40),
        goal_rate_threshold=0.5,
    ),
    CurriculumPhase(
        name="complex",
        geometry_tiers=(GeometryTier.TIER_3A, GeometryTier.TIER_3B),
        n_agents_range=(20, 60),
        goal_rate_threshold=0.4,
    ),
    CurriculumPhase(
        name="full",
        geometry_tiers=(
            GeometryTier.TIER_0,
            GeometryTier.TIER_1,
            GeometryTier.TIER_2,
            GeometryTier.TIER_3A,
            GeometryTier.TIER_3B,
        ),
        n_agents_range=(20, 100),
        goal_rate_threshold=0.0,
        tier_weights=(0.10, 0.15, 0.25, 0.25, 0.25),
    ),
)


@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum learning configuration.

    Success-rate-driven phase advancement per OpenAI ADR and Narvekar et al. (2020).
    """

    phases: tuple[CurriculumPhase, ...] = DEFAULT_CURRICULUM_PHASES
    """Ordered difficulty phases."""

    metric_window: int = 50
    """Number of recent episodes to average for promotion check."""

    min_episodes_per_phase: int = 200
    """Minimum episodes before phase advancement is allowed."""

    history_mix_ratio: float = 0.2
    """Fraction of episodes sampled from earlier phases to prevent forgetting
    (curriculum learning best practice, Narvekar et al. 2020)."""


@dataclass(frozen=True)
class VecEnvConfig:
    """Vectorized environment configuration."""

    n_envs: int = 4
    """Number of parallel environment workers."""

    n_steps_per_collect: int = 4096
    """Target agent-steps to collect before each PPO update.
    Larger values improve GPU utilization but increase memory."""


@dataclass(frozen=True)
class LogConfig:
    """Logging configuration."""

    backend: str = "tensorboard"
    """Logging backend: 'tensorboard' or 'wandb'."""

    log_dir: str = "runs"
    """Directory for TensorBoard logs."""

    log_interval: int = 1
    """Log metrics every N rollouts."""

    project_name: str = "crowdrl"
    """W&B project name (if using wandb backend)."""


@dataclass(frozen=True)
class DDPConfig:
    """Distributed Data Parallel configuration (single-node multi-GPU).

    Uses the DD-PPO pattern (Wijmans et al. 2019): each GPU rank collects
    rollouts independently, gradients are averaged via ``all_reduce``.

    Launch with ``torchrun --standalone --nproc_per_node=N script.py``.
    """

    backend: str = "nccl"
    """Communication backend. 'nccl' for GPU, 'gloo' for CPU."""


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training configuration. Composes all sub-configs."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    log: LogConfig = field(default_factory=LogConfig)
    env: CrowdEnvConfig = field(default_factory=CrowdEnvConfig)
    vec_env: VecEnvConfig = field(default_factory=VecEnvConfig)

    total_timesteps: int = 10_000_000
    """Total agent-steps to train for."""

    seed: int = 42
    """Random seed for reproducibility."""

    device: str = "cpu"
    """PyTorch device: 'cpu' or 'cuda'."""

    normalize_obs: bool = True
    """Running observation normalization. Andrychowicz et al. (2021):
    'Always use observation normalization.'"""

    normalize_rewards: bool = True
    """Reward normalization: divide by running std, no mean subtraction.
    Yu et al. (2022), Huang et al. (2022, detail #30)."""

    normalize_values: bool = True
    """Value target normalization (ValueNorm). Yu et al. (2022):
    'often helps and never hurts.'"""

    checkpoint_interval: int = 50
    """Save checkpoint every N rollouts."""

    checkpoint_dir: str = "checkpoints"
    """Directory for model checkpoints."""

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        # Convert GeometryTier enums to strings for JSON
        _convert_enums(d)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        """Deserialise from a dict (e.g., loaded from JSON).

        Handles nested frozen dataclasses and enum reconstruction.
        Only reconstructs top-level TrainConfig fields; nested env config
        fields are left as defaults since CrowdEnvConfig has its own
        frozen nested dataclasses that are complex to reconstruct from dicts.
        """
        from crowdrl_env.geometry_generator import GeometryConfig
        from crowdrl_env.reward import RewardConfig
        from crowdrl_env.spawner import SpawnConfig
        from crowdrl_core.observation import ObsConfig
        from crowdrl_core.action import ActionConfig
        from crowdrl_env.solvability import SolvabilityMode

        # Reconstruct CurriculumPhase objects with GeometryTier enums
        if "curriculum" in d:
            cur = d["curriculum"]
            if isinstance(cur, dict):
                if "phases" in cur:
                    phases = []
                    for p in cur["phases"]:
                        tiers = tuple(
                            GeometryTier[t] if isinstance(t, str) else GeometryTier(t)
                            for t in p["geometry_tiers"]
                        )
                        tw = p.get("tier_weights")
                        if tw is not None:
                            tw = tuple(tw)
                        phases.append(
                            CurriculumPhase(
                                name=p["name"],
                                geometry_tiers=tiers,
                                n_agents_range=tuple(p["n_agents_range"]),
                                goal_rate_threshold=p["goal_rate_threshold"],
                                tier_weights=tw,
                            )
                        )
                    cur["phases"] = tuple(phases)
                d["curriculum"] = CurriculumConfig(**cur)

        # Reconstruct nested frozen dataclasses
        if "network" in d and isinstance(d["network"], dict):
            net = d["network"]
            if "actor_hidden_sizes" in net:
                net["actor_hidden_sizes"] = tuple(net["actor_hidden_sizes"])
            if "critic_hidden_sizes" in net:
                net["critic_hidden_sizes"] = tuple(net["critic_hidden_sizes"])
            d["network"] = NetworkConfig(**net)
        if "ppo" in d and isinstance(d["ppo"], dict):
            d["ppo"] = PPOConfig(**d["ppo"])
        if "log" in d and isinstance(d["log"], dict):
            d["log"] = LogConfig(**d["log"])
        if "vec_env" in d and isinstance(d["vec_env"], dict):
            d["vec_env"] = VecEnvConfig(**d["vec_env"])

        # Reconstruct CrowdEnvConfig with its nested dataclasses
        if "env" in d and isinstance(d["env"], dict):
            env = d["env"]
            if "geometry" in env and isinstance(env["geometry"], dict):
                geo = env["geometry"]
                if "tier" in geo and isinstance(geo["tier"], str):
                    geo["tier"] = GeometryTier[geo["tier"]]
                # Convert tuple fields
                for field_name in (
                    "corridor_width_range",
                    "corridor_length_range",
                    "bottleneck_aperture_range",
                    "bottleneck_depth_range",
                    "branch_width_range",
                    "branch_length_range",
                ):
                    if field_name in geo and isinstance(geo[field_name], list):
                        geo[field_name] = tuple(geo[field_name])
                env["geometry"] = GeometryConfig(**geo)
            if "spawn" in env and isinstance(env["spawn"], dict):
                sp = env["spawn"]
                if "n_agents_range" in sp and isinstance(sp["n_agents_range"], list):
                    sp["n_agents_range"] = tuple(sp["n_agents_range"])
                env["spawn"] = SpawnConfig(**sp)
            if "obs" in env and isinstance(env["obs"], dict):
                obs_d = env["obs"]
                if "raycast" in obs_d and isinstance(obs_d["raycast"], dict):
                    from crowdrl_core.sensing import RaycastConfig

                    obs_d["raycast"] = RaycastConfig(**obs_d["raycast"])
                env["obs"] = ObsConfig(**obs_d)
            if "action" in env and isinstance(env["action"], dict):
                env["action"] = ActionConfig(**env["action"])
            if "reward" in env and isinstance(env["reward"], dict):
                env["reward"] = RewardConfig(**env["reward"])
            if "geometry_tiers" in env and env["geometry_tiers"] is not None:
                env["geometry_tiers"] = [
                    GeometryTier[t] if isinstance(t, str) else t for t in env["geometry_tiers"]
                ]
            if "solvability_mode" in env and isinstance(env["solvability_mode"], str):
                env["solvability_mode"] = SolvabilityMode[env["solvability_mode"]]
            d["env"] = CrowdEnvConfig(**env)

        return cls(**d)

    def save_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=_NumpyEncoder)

    @classmethod
    def load_json(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _convert_enums(d: dict) -> None:
    """Recursively convert enum values to their names for JSON serialisation."""
    for k, v in d.items():
        if isinstance(v, dict):
            _convert_enums(v)
        elif isinstance(v, (list, tuple)):
            converted = []
            for item in v:
                if isinstance(item, dict):
                    _convert_enums(item)
                    converted.append(item)
                elif hasattr(item, "name") and hasattr(item, "value"):
                    converted.append(item.name)
                else:
                    converted.append(item)
            d[k] = converted
        elif hasattr(v, "name") and hasattr(v, "value"):
            d[k] = v.name

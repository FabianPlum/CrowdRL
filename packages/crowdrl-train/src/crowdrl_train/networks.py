"""Actor-Critic networks for MAPPO with parameter sharing.

Architecture decisions grounded in:
- Andrychowicz et al. (2021): separate actor/critic, tanh activation, 2 hidden layers.
- Huang et al. (2022): orthogonal init (sqrt(2) hidden, 0.01 actor out, 1.0 critic out).
- Yu et al. (2022): state-independent log_std, diagonal Gaussian for continuous control.

All agents share one actor and one critic (parameter sharing). Agent heterogeneity
(body size, preferred speed) enters through the observation, not separate networks.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from crowdrl_train.config import NetworkConfig


def _ortho_init(weight: torch.Tensor, gain: float = 1.0) -> None:
    """Orthogonal initialization via numpy QR decomposition.

    Avoids torch.nn.init.orthogonal_ which can crash on Windows due to
    LAPACK access violations in certain PyTorch builds.

    Produces the same result: a (semi-)orthogonal matrix scaled by gain.
    """
    rows, cols = weight.shape[0], np.prod(weight.shape[1:])
    flat = np.random.randn(rows, cols).astype(np.float32)
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    # Make Q uniform (remove sign ambiguity)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    q = q[: weight.shape[0], : np.prod(weight.shape[1:])]
    with torch.no_grad():
        weight.copy_(torch.from_numpy(gain * q.reshape(weight.shape)))


def _make_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    activation: str,
    ortho_init: bool,
) -> nn.Sequential:
    """Build an MLP with the specified hidden layers and activation."""
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        linear = nn.Linear(prev, h)
        if ortho_init:
            _ortho_init(linear.weight, gain=math.sqrt(2))
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(act_cls())
        prev = h
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network: obs → action mean + state-independent log_std.

    The action distribution is a diagonal Gaussian with learnable but
    state-independent standard deviation — the standard approach for
    on-policy continuous control (Huang et al. 2022, detail #24).
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.feature_net = _make_mlp(
            config.obs_dim, config.actor_hidden_sizes, config.activation, config.ortho_init
        )
        last_hidden = config.actor_hidden_sizes[-1]
        self.action_mean = nn.Linear(last_hidden, config.action_dim)
        if config.ortho_init:
            # Small initial weights → near-zero initial actions (Andrychowicz et al.)
            _ortho_init(self.action_mean.weight, gain=0.01)
            nn.init.zeros_(self.action_mean.bias)

        # State-independent log_std, init to log(0.5) ≈ -0.693
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init))

    def forward(self, obs: torch.Tensor) -> Normal:
        """Return the action distribution for the given observations.

        Parameters
        ----------
        obs : (batch, obs_dim) tensor

        Returns
        -------
        Normal distribution with shapes (batch, action_dim)
        """
        features = self.feature_net(obs)
        mean = self.action_mean(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, entropy).

        Actions are clipped to [-1, 1] for the environment, but log_prob
        is computed from the unclipped sample (Huang et al. 2022, detail #27).
        """
        dist = self.forward(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        # Log-prob of the raw (unclipped) sample — summed across action dims
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Clip for the environment, but log_prob uses the raw sample
        action = raw_action.clamp(-1.0, 1.0)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log_prob and entropy for stored (unclipped) actions.

        Used during PPO update to compute the importance sampling ratio.
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Value network: obs (+ optional global context) → scalar value.

    Separate from actor per Andrychowicz et al. (2021). In CTDE mode
    (Yu et al. 2022), the critic receives obs + compact global context
    while the actor sees only local observations.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        critic_input_dim = config.critic_obs_dim or config.obs_dim
        self.feature_net = _make_mlp(
            critic_input_dim, config.critic_hidden_sizes, config.activation, config.ortho_init
        )
        last_hidden = config.critic_hidden_sizes[-1]
        self.value_head = nn.Linear(last_hidden, 1)
        if config.ortho_init:
            _ortho_init(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimate.

        Parameters
        ----------
        obs : (batch, obs_dim) or (batch, critic_obs_dim) tensor

        Returns
        -------
        (batch, 1) value tensor
        """
        features = self.feature_net(obs)
        return self.value_head(features)


class ActorCritic(nn.Module):
    """Convenience wrapper holding both actor and critic.

    This is not a shared-trunk architecture — the actor and critic have
    fully independent parameters. This wrapper provides a single object
    for checkpointing and device management.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.actor = Actor(config)
        self.critic = Critic(config)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for rollout collection.

        Returns (action, raw_action, log_prob, entropy, value).
        raw_action is the unclipped sample stored for PPO re-evaluation.
        """
        dist = self.actor(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        action = raw_action.clamp(-1.0, 1.0)

        value = self.critic(critic_obs if critic_obs is not None else obs)
        return action, raw_action, log_prob, entropy, value.squeeze(-1)

    def get_value(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Value-only forward pass (for GAE bootstrap)."""
        value = self.critic(critic_obs if critic_obs is not None else obs)
        return value.squeeze(-1)

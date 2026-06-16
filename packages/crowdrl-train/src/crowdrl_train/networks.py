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

# Hard bounds on the policy log-std, applied every forward pass, as a backstop.
# The primary fix for the std-runaway collapse is the tanh action squash below:
# the bounded action space no longer rewards inflating std to reach a hard-clamp
# boundary (the old failure mode -- std drifting up until actions were saturated
# bang-bang noise -> mass collisions -> abrupt collapse). These bounds remain a
# safety net: the upper bound caps the pre-squash std at exp(0.0) = 1.0; the
# lower bound keeps a floor of exp(-5) ~ 0.007 so exploration can't collapse.
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 0.0


def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    """Log-prob of the tanh-squashed action ``a = tanh(raw_action)``.

    For a diagonal Gaussian base and an element-wise tanh squash, the
    change-of-variables gives, summed over action dims,

        log pi(a) = sum_i [ log N(u_i; mu_i, sigma_i) - log(1 - tanh(u_i)^2) ]

    The Jacobian term uses the numerically stable identity

        log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))

    to avoid catastrophic cancellation / -inf when |u| is large (exactly the
    regime that triggered the prior std-runaway collapse). ``raw_action`` is the
    pre-squash sample ``u`` (Gaussian space), stored at collection time so the
    PPO ratio re-evaluation stays consistent with the squashed policy.
    """
    base = dist.log_prob(raw_action)  # (..., action_dim)
    correction = 2.0 * (math.log(2.0) - raw_action - nn.functional.softplus(-2.0 * raw_action))
    return (base - correction).sum(dim=-1)


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
        # Clamp log_std to [_LOG_STD_MIN, _LOG_STD_MAX] every forward pass so the
        # learnable std can neither run away (collapse trigger) nor collapse to 0.
        std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp().expand_as(mean)
        return Normal(mean, std)

    def current_std(self) -> torch.Tensor:
        """Per-dim action std actually in effect (post-clamp), for logging.

        Exposes the policy's exploration scale so the training loop can watch
        for std drift directly instead of inferring it from the entropy curve.
        """
        return self.log_std.detach().clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, entropy).

        The action is tanh-squashed into (-1, 1); ``log_prob`` includes the
        tanh change-of-variables correction (computed from the pre-squash
        sample ``u``). The deterministic action is ``tanh(mean)``. ``entropy``
        is the base-Gaussian entropy, used only for the exploration bonus.
        """
        dist = self.forward(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        log_prob = _squashed_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)

        action = torch.tanh(raw_action)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log_prob and entropy for stored pre-squash actions.

        ``actions_raw`` are the pre-tanh samples ``u`` stored at collection
        time; the tanh-squashed log-prob is recomputed from them so the PPO
        importance ratio is consistent with the squashed policy.
        """
        dist = self.forward(obs)
        log_prob = _squashed_log_prob(dist, actions_raw)
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

    Note on "MAPPO": this is parameter-shared PPO -- a single actor and a single
    critic shared across all agents, with each critic call seeing only that
    agent's LOCAL observation. It is not centralized-critic CTDE MAPPO (the critic
    has no global-state input by default; see ``NetworkConfig.critic_obs_dim`` for
    the CTDE hook).
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
        ``action`` is the tanh-squashed action sent to the env; ``raw_action``
        is the pre-squash sample ``u`` (Gaussian space) stored for PPO
        re-evaluation. ``log_prob`` includes the tanh Jacobian correction.
        """
        dist = self.actor(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        log_prob = _squashed_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        action = torch.tanh(raw_action)

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

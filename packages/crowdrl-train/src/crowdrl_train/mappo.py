"""MAPPO update: PPO with parameter sharing for multi-agent continuous control.

Implementation follows:
- Yu et al. (2022): full-batch update (n_minibatches=1), KL early stopping,
  permissive gradient clipping (max_grad_norm=10.0).
- Andrychowicz et al. (2021): MSE value loss (no clipping), separate actor/critic.
- Huang et al. (2022): advantage normalization, cosine/linear LR decay.

The update operates on a FlatBatch of active agent-steps from the RolloutBuffer.
Since all agents share one policy (parameter sharing), each agent-step is an
independent sample — the standard PPO update applies directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from crowdrl_train.buffer import FlatBatch
from crowdrl_train.config import PPOConfig
from crowdrl_train.networks import ActorCritic


class MAPPOUpdater:
    """Runs PPO update epochs on a collected rollout batch.

    When *distributed* is ``True`` (or auto-detected from
    ``torch.distributed``), gradients are averaged across ranks via
    ``allreduce_gradients`` after each ``backward()`` call, before
    the optimiser step.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: PPOConfig,
        device: torch.device,
        distributed: bool | None = None,
    ):
        self.actor_critic = actor_critic
        self.config = config
        self.device = device

        # Auto-detect distributed when not explicitly set
        if distributed is None:
            self._distributed = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
        else:
            self._distributed = distributed

        # Separate optimisers for actor and critic (can have different LR)
        self.actor_optimizer = torch.optim.Adam(
            actor_critic.actor.parameters(),
            lr=config.lr_actor,
            eps=config.adam_eps,
        )
        self.critic_optimizer = torch.optim.Adam(
            actor_critic.critic.parameters(),
            lr=config.lr_critic,
            eps=config.adam_eps,
        )

        self._rng = torch.Generator(device="cpu")

    def update(self, batch: FlatBatch) -> dict[str, float]:
        """Run PPO update epochs on the flat batch.

        Returns a metrics dict with:
            policy_loss, value_loss, entropy, approx_kl,
            clip_fraction, explained_variance
        """
        cfg = self.config

        # Normalise advantages per-batch (Yu et al. 2022)
        if cfg.normalize_advantages and batch.batch_size > 1:
            adv = batch.advantages
            batch = FlatBatch(
                obs=batch.obs,
                actions_raw=batch.actions_raw,
                log_probs=batch.log_probs,
                advantages=(adv - adv.mean()) / (adv.std() + 1e-8),
                returns=batch.returns,
                values=batch.values,
            )

        # Accumulate metrics across epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        n_updates = 0
        early_stopped = False

        for epoch in range(cfg.n_epochs):
            if early_stopped:
                break

            mb_indices = batch.minibatch_indices(cfg.n_minibatches, self._rng)

            for indices in mb_indices:
                mb_obs = batch.obs[indices]
                mb_actions_raw = batch.actions_raw[indices]
                mb_old_log_probs = batch.log_probs[indices]
                mb_advantages = batch.advantages[indices]
                mb_returns = batch.returns[indices]
                mb_old_values = batch.values[indices]

                # Re-evaluate actions under current policy
                new_log_probs, entropy = self.actor_critic.actor.evaluate_actions(
                    mb_obs, mb_actions_raw
                )
                new_values = self.actor_critic.get_value(mb_obs)

                # --- Policy loss (clipped surrogate) ---
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # Approximate KL for early stopping (Huang et al. 2022, detail #14)
                with torch.no_grad():
                    approx_kl_t = ((ratio - 1) - log_ratio).mean()
                    # CRITICAL: under DDP, all ranks must agree on the early-stop
                    # decision -- otherwise one rank exits the loop and proceeds
                    # to the next collective (e.g. normalizer sync) while another
                    # rank is still issuing gradient all-reduces, causing NCCL
                    # to deadlock on mismatched collectives.
                    # Averaging KL across ranks also matches the global-batch
                    # semantics: we stop when the global KL exceeds threshold,
                    # not when any single rank's local KL does.
                    if self._distributed:
                        torch.distributed.all_reduce(
                            approx_kl_t, op=torch.distributed.ReduceOp.SUM
                        )
                        approx_kl_t.div_(torch.distributed.get_world_size())
                    approx_kl = approx_kl_t.item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item()

                # KL early stopping (Yu et al. 2022) -- now using global KL
                # under DDP so all ranks agree on the decision.
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    early_stopped = True
                    break

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss ---
                if cfg.use_value_clip:
                    # PPO-style value clipping (OFF by default per Andrychowicz et al.)
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -cfg.clip_epsilon,
                        cfg.clip_epsilon,
                    )
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                elif cfg.use_huber_loss:
                    value_loss = nn.functional.huber_loss(
                        new_values, mb_returns, delta=cfg.huber_delta
                    )
                else:
                    # MSE — the default (Andrychowicz et al. 2021)
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                # --- Actor update ---
                actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self._distributed:
                    _allreduce_gradients(self.actor_critic.actor)
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                # --- Critic update ---
                critic_loss = cfg.value_coef * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self._distributed:
                    _allreduce_gradients(self.actor_critic.critic)
                nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                n_updates += 1

        # Explained variance
        with torch.no_grad():
            y_pred = batch.values
            y_true = batch.returns
            var_y = y_true.var()
            ev = 1 - (y_true - y_pred).var() / (var_y + 1e-8) if var_y > 1e-8 else 0.0
            if isinstance(ev, torch.Tensor):
                ev = ev.item()

        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "explained_variance": ev,
            "n_epochs_actual": n_updates / max(cfg.n_minibatches, 1),
        }

    def update_learning_rate(self, progress: float) -> None:
        """Update learning rate based on training progress.

        Parameters
        ----------
        progress : float in [0, 1] — fraction of total training complete
        """
        if self.config.lr_schedule == "cosine":
            frac = 0.5 * (1.0 + math.cos(math.pi * progress))
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = self.config.lr_actor * frac
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.config.lr_critic * frac
        elif self.config.lr_schedule == "linear":
            frac = 1.0 - progress
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = self.config.lr_actor * frac
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.config.lr_critic * frac


def _allreduce_gradients(model: nn.Module) -> None:
    """Average gradients across all DDP ranks via a single all-reduce.

    Flattens every ``.grad`` into one contiguous buffer so only one
    NCCL collective is issued per model (actor or critic).
    """
    grads: list[torch.Tensor] = []
    params_with_grad: list[nn.Parameter] = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
            params_with_grad.append(p)

    if not grads:
        return

    flat = torch.cat(grads)
    torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
    flat.div_(torch.distributed.get_world_size())

    offset = 0
    for p in params_with_grad:
        numel = p.grad.numel()
        p.grad.copy_(flat[offset : offset + numel].view_as(p.grad))
        offset += numel

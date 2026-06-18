"""NaN/Inf containment (H4): a transient non-finite must not kill the run.

Two chokepoints:
  * TorchRunningNormalizer.update must drop non-finite samples (a poisoned
    running mean/var would NaN every future normalized obs -> dead policy).
  * batched_step must keep agent state (and the obs built from it) finite even
    if a force/velocity goes non-finite in a degenerate pileup.
"""

from __future__ import annotations

import torch

from crowdrl_torch.normalizer import TorchRunningNormalizer
from crowdrl_torch.step import batched_step
from crowdrl_torch.types import EnvConfig, make_initial_state

INF = float("inf")
NAN = float("nan")


# --- normalizer poison-proofing ------------------------------------------------


def test_update_ignores_nonfinite_rows():
    norm = TorchRunningNormalizer(shape=(3,), device="cpu")
    good = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    bad = torch.tensor([[NAN, 0.0, 0.0], [INF, 0.0, 0.0]])
    norm.update(torch.cat([good, bad], dim=0))
    assert torch.isfinite(norm.mean).all()
    assert torch.isfinite(norm.var).all()
    # Stats reflect only the finite rows -> mean ~ [1, 2, 3] (small bias from the
    # normalizer's 1e-4 initial pseudo-count).
    assert torch.allclose(norm.mean, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), atol=1e-3)


def test_all_nonfinite_batch_is_noop():
    norm = TorchRunningNormalizer(shape=(2,), device="cpu")
    mean0, var0, count0 = norm.mean.clone(), norm.var.clone(), norm.count
    norm.update(torch.tensor([[NAN, INF], [-INF, NAN]]))
    assert torch.equal(norm.mean, mean0)
    assert torch.equal(norm.var, var0)
    assert norm.count == count0


def test_normalize_finite_after_poison_attempt():
    norm = TorchRunningNormalizer(shape=(2,), device="cpu")
    norm.update(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    norm.update(torch.tensor([[NAN, 5.0]]))  # poison attempt -> ignored
    out = norm.normalize(torch.tensor([[2.0, 2.0]]))
    assert torch.isfinite(out).all()


# --- batched_step finite-state invariant --------------------------------------


def _toy_state(n=4):
    state = make_initial_state(
        n_envs=1,
        max_agents=n,
        max_segments=4,
        max_waypoints=4,
        memory_window=5,
        k_neighbours=4,
        neighbor_vel_history_window=2,
        device="cpu",
    )
    state.positions[0] = torch.tensor([[1.0, 1.0], [1.2, 1.0], [5.0, 5.0], [0.0, 0.0]])
    state.goal_positions[:] = torch.tensor([5.0, 5.0])
    state.shoulder_widths[:] = 0.25
    state.chest_depths[:] = 0.15
    state.preferred_speeds[:] = 1.3
    state.active_mask[0] = torch.tensor([True, True, True, False])
    state.n_agents[:] = 3
    state.wall_segments[0, 0] = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    state.n_segments[:] = 1
    return state


def _toy_config(n=4):
    return EnvConfig(
        max_agents=n,
        max_segments=4,
        max_waypoints=4,
        n_rays=8,
        k_neighbours=4,
        use_navmesh=False,
        use_temporal_memory=False,
        use_neighbor_memory=False,
        temporal_memory_window=5,
        neighbor_vel_history_window=2,
    )


def test_step_recovers_from_nonfinite_input_velocity():
    """A transient NaN/Inf velocity must not propagate into the next state/obs."""
    state = _toy_state()
    state.velocities[0, 0] = torch.tensor([NAN, INF])  # injected glitch
    new_state, obs, rewards, _term, _trunc, _comps = batched_step(
        state, torch.zeros((1, 4, 4)), _toy_config()
    )
    assert torch.isfinite(new_state.velocities).all()
    assert torch.isfinite(new_state.positions).all()
    assert torch.isfinite(obs).all()
    assert torch.isfinite(rewards).all()


def test_step_finite_under_coincident_pileup():
    """Near-coincident agents driven at full action stay finite."""
    state = _toy_state()
    state.positions[0, 1] = torch.tensor([1.0, 1.0])  # exactly coincident with agent 0
    new_state, obs, rewards, _term, _trunc, _comps = batched_step(
        state, torch.full((1, 4, 4), 1.0), _toy_config()
    )
    assert torch.isfinite(new_state.positions).all()
    assert torch.isfinite(new_state.velocities).all()
    assert torch.isfinite(obs).all()
    assert torch.isfinite(rewards).all()


def test_step_velocity_weighted_collision_wires_pre_contact_snapshot():
    """End-to-end through batched_step: the impact-speed-weighted collision
    penalty (P1) must run (finite rewards) and actually reshape the reward
    relative to the binary penalty. Agents 0,1 sit at contact distance, so the
    collision term is exercised and the pre-contact velocity snapshot is wired
    into compute_rewards."""
    actions = torch.zeros((1, 4, 4))
    actions[0, :, 0] = 1.0  # drive forward

    base = _toy_config()
    weighted = base._replace(
        collision_penalty=-2.0,
        use_velocity_weighted_collision=True,
        collision_speed_floor=0.5,
        collision_speed_scale=0.5,
    )
    binary = base._replace(collision_penalty=-2.0)

    _, _, r_weighted, _, _, _ = batched_step(_toy_state(), actions, weighted)
    _, _, r_binary, _, _, _ = batched_step(_toy_state(), actions, binary)

    assert torch.isfinite(r_weighted).all()
    # The reshaping (floor 0.5 + closing-speed scale) must change the reward
    # stream relative to the flat binary penalty.
    assert not torch.allclose(r_weighted, r_binary)


def test_step_velocity_weighted_finite_under_nonfinite_velocity():
    """The r855 NaN bug: with the velocity-weighted penalties ON, a transient
    non-finite velocity fed the unguarded pre-contact snapshot into the reward,
    producing a NaN that poisoned training. The reward must now sanitize the
    snapshot so the reward stays finite even when a velocity goes NaN/Inf."""
    state = _toy_state()
    state.velocities[0, 0] = torch.tensor([NAN, INF])  # injected glitch
    cfg = _toy_config()._replace(
        use_velocity_weighted_collision=True,
        use_velocity_weighted_proximity=True,
        collision_speed_floor=0.1,
        collision_speed_scale=0.5,
        proximity_speed_floor=0.0,
        proximity_speed_scale=0.5,
    )
    new_state, obs, rewards, _term, _trunc, _comps = batched_step(
        state, torch.full((1, 4, 4), 1.0), cfg
    )
    assert torch.isfinite(rewards).all()
    assert torch.isfinite(new_state.velocities).all()
    assert torch.isfinite(obs).all()

"""GAE bootstrap semantics for the GPU rollout collector.

Guards the time-limit fix (H3): truncation (timeout/stuck) must bootstrap the
agent's own value V(s_t) as a proxy for V(s_{t+1}), whereas termination (goal)
is a true terminal and zeroes the bootstrap. Treating truncation as a hard
terminal systematically under-values states near the time limit.
"""

from __future__ import annotations

import numpy as np

from crowdrl_torch.torch_collector import TorchRolloutCollector

GAMMA = 0.99
LAM = 0.95


def _gae(active, rew, val, term, trunc, bootstrap):
    """Thin wrapper: wrap scalars-per-step into (N=1,) arrays and call the helper."""
    ep_active = [np.array([a], dtype=np.float64) for a in active]
    ep_rew = [np.array([r], dtype=np.float64) for r in rew]
    ep_val = [np.array([v], dtype=np.float64) for v in val]
    ep_term = [np.array([t], dtype=np.bool_) for t in term]
    ep_trunc = [np.array([t], dtype=np.bool_) for t in trunc]
    adv, ret = TorchRolloutCollector._segment_gae(
        ep_active,
        ep_rew,
        ep_val,
        ep_term,
        ep_trunc,
        np.array([bootstrap], dtype=np.float64),
        GAMMA,
        LAM,
    )
    return np.array([a[0] for a in adv]), np.array([r[0] for r in ret])


def test_terminated_step_zeroes_bootstrap():
    """A goal step is a true terminal: return == reward, advantage == r - V."""
    adv, ret = _gae(active=[1.0], rew=[2.0], val=[5.0], term=[True], trunc=[False], bootstrap=0.0)
    assert np.allclose(ret, [2.0])
    assert np.allclose(adv, [2.0 - 5.0])


def test_truncated_step_bootstraps_own_value():
    """A timeout/stuck step bootstraps V(s_t): return == r + gamma * V(s_t)."""
    adv, ret = _gae(active=[1.0], rew=[2.0], val=[5.0], term=[False], trunc=[True], bootstrap=0.0)
    assert np.allclose(ret, [2.0 + GAMMA * 5.0])
    assert np.allclose(adv, [2.0 + GAMMA * 5.0 - 5.0])


def test_truncation_differs_from_termination():
    """The whole point of H3: same transition, different bootstrap by flag."""
    _, ret_term = _gae([1.0], [2.0], [5.0], [True], [False], 0.0)
    _, ret_trunc = _gae([1.0], [2.0], [5.0], [False], [True], 0.0)
    assert ret_trunc[0] > ret_term[0]  # truncation keeps future value, termination doesn't
    assert np.allclose(ret_trunc[0] - ret_term[0], GAMMA * 5.0)


def test_trailing_segment_bootstraps_from_critic():
    """An unfinished (no-done) trailing step bootstraps the supplied critic value."""
    adv, ret = _gae(
        active=[1.0], rew=[1.0], val=[4.0], term=[False], trunc=[False], bootstrap=10.0
    )
    # return = r + gamma * V_bootstrap
    assert np.allclose(ret, [1.0 + GAMMA * 10.0])


def test_inactive_steps_zeroed():
    """Inactive (post-done) steps contribute zero advantage/return delta."""
    adv, ret = _gae(
        active=[1.0, 0.0],
        rew=[1.0, 0.0],
        val=[4.0, 0.0],
        term=[True, False],
        trunc=[False, False],
        bootstrap=0.0,
    )
    assert np.allclose(adv[1], 0.0)
    # First (active, terminated) step: return == reward.
    assert np.allclose(ret[0], 1.0)

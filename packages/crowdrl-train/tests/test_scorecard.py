"""Tests for the fixed-scenario behavioural scorecard.

The runner is exercised on a random (untrained) policy via the shared conftest
fixtures, so no checkpoint is needed; correctness of the underlying metrics lives
in crowdrl-env's test_eval_metrics.py.
"""

from __future__ import annotations

from crowdrl_env.geometry_generator import GeometryTier

from crowdrl_train.scorecard import (
    DEFAULT_SCENARIOS,
    ScenarioSpec,
    format_scorecard,
    run_scorecard_policy,
)


def test_default_scenarios_are_well_formed():
    assert len(DEFAULT_SCENARIOS) >= 4
    assert all(isinstance(s, ScenarioSpec) for s in DEFAULT_SCENARIOS)
    assert all(s.n_agents > 0 for s in DEFAULT_SCENARIOS)
    # Spans more than one tier so the scorecard probes both failure modes.
    assert len({s.tier for s in DEFAULT_SCENARIOS}) >= 3


def test_run_scorecard_policy_smoke(tiny_actor_critic, tiny_env_config):
    # One tiny Tier-0 scenario, short episode -> exercises the full
    # rollout -> metrics -> aggregation path end to end.
    scenarios = [ScenarioSpec("t0_smoke", GeometryTier.TIER_0, 4, 0)]
    sc = run_scorecard_policy(
        tiny_env_config, tiny_actor_critic, None, scenarios=scenarios, max_steps=12
    )
    assert len(sc["per_scenario"]) == 1
    entry = sc["per_scenario"][0]
    assert entry["label"] == "t0_smoke"
    assert entry["n_agents"] == 4
    assert "goal_rate" in entry["metrics"]
    assert "freeze_rate" in entry["metrics"]
    assert "goal_rate" in sc["overall"]


def test_run_scorecard_policy_is_deterministic(tiny_actor_critic, tiny_env_config):
    # Deterministic (mean) actions + fixed scenario seed -> identical metrics.
    scenarios = [ScenarioSpec("t0", GeometryTier.TIER_0, 4, 0)]
    a = run_scorecard_policy(
        tiny_env_config, tiny_actor_critic, None, scenarios=scenarios, max_steps=12
    )
    b = run_scorecard_policy(
        tiny_env_config, tiny_actor_critic, None, scenarios=scenarios, max_steps=12
    )
    assert a["per_scenario"][0]["metrics"] == b["per_scenario"][0]["metrics"]


def test_format_scorecard_renders_table():
    sc = {
        "per_scenario": [
            {
                "label": "open_t0",
                "n_agents": 20,
                "seed": 0,
                "metrics": {
                    "goal_rate": 0.9,
                    "agent_collision_rate": 0.1,
                    "freeze_rate": 0.05,
                },
            },
        ],
        "overall": {"goal_rate": 0.9, "agent_collision_rate": 0.1, "freeze_rate": 0.05},
    }
    out = format_scorecard(sc)
    assert "scenario" in out
    assert "open_t0" in out
    assert "OVERALL" in out
    assert "goal" in out  # a column header
    # A metric absent from the dict (e.g. wall) renders as "--", not a KeyError.
    assert "--" in out

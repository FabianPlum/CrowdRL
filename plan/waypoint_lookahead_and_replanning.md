# Waypoint look-ahead & dynamic re-planning — idea capture

> **Status: SHELVED -- not currently planned (2026-07-30).** The JuPedSim
> integration rules this out: the operational-model layer is handed only the
> single routed next waypoint (`ped.next_target`) -- neither the remaining
> route (avenue A's lookahead) nor control over re-planning (avenue B; the
> router replans upstream of the operational step, outside the model) is
> available to a deployed model. Training on either would violate "do not
> train on a signal deployment cannot supply", the principle behind the
> `use_jupedsim_style_routing` contract that produced the shipped r0125 line.
> Revisit only if the upstream routing surface grows (cf. the radius-aware
> router-inset feature request, which would be co-drafted upstream).
> *(Original status, 2026-06-18: future work, out of scope for PR #4; idea
> preserved on branch `plan/waypoint-lookahead-idea`.)*

## Problem

With the current single next-waypoint navigation signal (navmesh-only mode,
`use_goal_direction: false`), agents frequently **overshoot the next
waypoint**. Overshoot can carry an agent to a position from which reaching
that waypoint is non-trivial — in many cases the waypoint ends up *behind*
the agent, so it must reverse or loop back to recover.

This is worst in exactly the geometries we care about:

- scenes with **many obstacles**, where a wrong move strands the agent in a
  pocket that is awkward to escape, and
- goals (or waypoints) inside a **narrow corridor** the agent can "zoom
  past" before it has slowed/turned, after which re-entry is hard.

The robust waypoint cursor from `42f5f93` (advance when closer to the next
waypoint) mitigates *mild* overshoot, but it does not recover an agent that
has overshot into a genuinely bad region, nor does it give the agent any
*advance warning* that a sharp turn is coming.

## Two avenues (in suggested trial order)

### A. Expose the next N waypoints (preferred first trial)

Instead of feeding only the single next-waypoint relative location, expose
the next **N** waypoints to the agent — as **N discrete relative vectors**,
**not** a smoothed/blended combination (we deliberately do not want to go
back to the pre-`fa18862` smoothed signal).

Rationale: this mirrors how humans motion-plan — we look several steps ahead
and pre-adjust. With a few waypoints of look-ahead the policy has the
information to produce **anticipatory behaviour**: when a (sharp) turn is
coming up soon it can **re-weigh its current speed** (slow down before the
turn) instead of overshooting and recovering. This pairs naturally with the
speed-turn coupling already in the action interpreter.

**Why this is the cheap first trial:** the full funnel path is *already
stored per agent* — `ObsConfig.navmesh_max_waypoints` (1024) with the GPU
waypoint buffer (`EnvConfig.max_waypoints`), advanced by the cursor in the
torch step. So exposing N waypoints is primarily an **observation-builder
change**: read N entries from the cursor forward, transform each into the
ego frame (same as the current single waypoint), pad/clamp near the goal
when fewer than N remain. No new path computation, no env-dynamics change.

Touch points to scope later:
- `packages/crowdrl-core/src/crowdrl_core/observation.py` — the navmesh
  block (currently `next_waypoint_direction` (2) + `path_deviation` (1));
  add an N-waypoint lookahead block behind an ablation flag.
- `packages/crowdrl-core/src/crowdrl_core/navmesh.py` — a helper returning
  the next N funnel waypoints from a cursor position (the funnel already
  produces the full list).
- `packages/crowdrl-torch/` (`observation.py`, `step.py`,
  `episode_factory.py`) — vectorised N-waypoint gather from the stored
  buffer + cursor; keep core/torch parity (cf. `test_navmesh_parity`).

Open questions:
- N = ? (start small, e.g. 2–4). Fixed count vs **arc-length-spaced**
  look-ahead (waypoints can be very unevenly spaced; a fixed count of raw
  funnel vertices may look only a few cm ahead on a dense path).
- Representation: relative position vs direction+distance; per-waypoint
  normalisation; how to encode "no more waypoints / near goal" padding.
- obs_dim cost (+2N) and its effect on the first-layer compression ratio
  (relevant given the A++ underfit concern).
- Ablation flag (`observation.use_waypoint_lookahead` + `n_waypoints`),
  default off, train == deploy parity for the JuPedSim adapter.
- Interaction with `use_goal_direction`: this is meant to *strengthen* the
  navmesh-only mode, not reintroduce the global bearing.

### B. Dynamic shortest-path re-planning (bigger, second)

When an agent has overshot / deviated, **re-compute its updated shortest
path to the goal** from the current position, so a new valid route is found
and the agent recovers instead of fighting a stale path.

- Trigger: the existing `path_deviation` scalar (already computed) crossing
  a threshold, or the cursor failing to advance for K steps.
- Mechanism: re-run A* + funnel (Simple Stupid Funnel) from the agent's
  current position to the goal and replace the stored waypoint buffer +
  reset the cursor. The navmesh and funnel code already exist; the work is
  doing this **selectively and cheaply at scale** (per-agent, on GPU /
  batched, only for flagged agents) without stalling the step.
- Risks/cost: re-planning is more expensive than a lookup; needs to be rare
  and batched. Must stay deterministic enough for train == deploy.

A is the lower-risk lever and may reduce overshoot enough that B is rarely
needed; B is the proper recovery mechanism for the hard obstacle / narrow-
corridor cases.

## Relationship to existing work

- Extends the single next-waypoint signal (`fa18862`) and complements the
  robust cursor (`42f5f93`).
- Explicitly **not** a return to the old smoothed-combination signal.
- Validate on sharp-turn and narrow-corridor scenarios (cf. the `sharpturn`
  configs and the Tier-1 bottleneck / Tier-2 L-bend generators).

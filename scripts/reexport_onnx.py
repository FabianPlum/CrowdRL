"""Re-export a trained checkpoint as a self-describing ONNX artefact.

Rebuilds the actor + frozen observation normalizer from a results dir's
``checkpoint_rollout_*.pt`` and its ``config_resolved.yaml`` (the config
contract), then exports through ``crowdrl_train.export.export_onnx`` so the
resolved obs/action configs and provenance land in the file's
``metadata_props`` (issue #7). The result deploys in crowdrl-jupedsim with
zero hand-supplied configuration.

Dynamics need explicit handling. The schema-v2 metadata block certifies four
env-level physics constants as the ones the policy trained under, but
``config_resolved.yaml`` can express only ``desired_velocity_weight``
(``train_mappo.build_env_config`` parses no other dynamics key, and
``cfg_dict_from_env_config`` emits no other). The remaining three would
otherwise be silently filled from *present-day* ``CrowdEnvConfig`` defaults
and stamped as the run's trained physics -- exactly the drift schema v2 exists
to prevent. This script therefore refuses to guess: every field the YAML does
not record must be supplied explicitly (or waived with
``--assume-current-defaults``), and the per-field origin is stamped into the
artefact's provenance as ``dynamics_provenance``.

Typical use -- export a checkpoint as the shipped example model (how
``example_model/policy_r0125.onnx`` was produced):

    uv run python scripts/reexport_onnx.py results_exp_jps_routing_ft_r0400 \
        checkpoint_rollout_0125.pt --output example_model/policy_r0125.onnx \
        --max-velocity-magnitude 3.0 \
        --contact-stiffness 30000 --contact-damping 500

Add ``--verify-against <existing.onnx>`` when upgrading a legacy artefact in
place, to prove the new export produces identical outputs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowdrl_train.export import export_onnx, verify_onnx  # noqa: E402
from crowdrl_train.networks import ActorCritic  # noqa: E402
from crowdrl_train.normalizer import RunningNormalizer  # noqa: E402

from train_mappo import _git_rev, build_env_config, build_net_config, load_config  # noqa: E402

DYNAMICS_FIELDS = (
    "desired_velocity_weight",
    "max_velocity_magnitude",
    "contact_stiffness",
    "contact_damping",
)

# Top-level YAML keys build_env_config actually reads for each dynamics field.
# Only desired_velocity_weight has one (train_mappo.py:189); the other three
# have no representation in the YAML schema at all.
DYNAMICS_YAML_KEYS = {"desired_velocity_weight": "desired_velocity_weight"}

# Keys from superseded YAML schemas that encoded dynamics today's parser
# ignores. Their presence proves the run predates the current formulation, so
# the corresponding field must be stated explicitly rather than defaulted.
STALE_YAML_KEYS = {"max_speed_multiplier": "max_velocity_magnitude"}


class DynamicsProvenanceError(RuntimeError):
    """The trained dynamics cannot be established from the available inputs."""


def resolve_dynamics(
    cfg: dict,
    env_config,
    overrides: dict[str, float | None],
    *,
    assume_current_defaults: bool = False,
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve the four dynamics fields, recording where each value came from.

    Parameters
    ----------
    cfg : the raw ``config_resolved.yaml`` dict (key presence is the evidence
        that a field was actually recorded for this run)
    env_config : the ``CrowdEnvConfig`` built from ``cfg``; supplies a value
        only for fields the YAML records or the caller waives
    overrides : per-field explicit values from the CLI (``None`` = not given)
    assume_current_defaults : accept present-day ``CrowdEnvConfig`` defaults
        for fields the YAML does not record, stamping them ``assumed-default``

    Returns
    -------
    ``(dynamics, provenance)`` -- the metadata block and a per-field origin map
    (``config_resolved.yaml`` / ``explicit`` / ``assumed-default``).

    Raises
    ------
    DynamicsProvenanceError
        If a field is neither recorded nor supplied and defaults are not
        waived, if an explicit value contradicts the YAML, or if a stale YAML
        key shows the run's value cannot be the current default.
    """
    dynamics: dict[str, float] = {}
    provenance: dict[str, str] = {}
    unresolved: list[str] = []

    stale_present = {field: key for key, field in STALE_YAML_KEYS.items() if key in cfg}

    for field in DYNAMICS_FIELDS:
        yaml_key = DYNAMICS_YAML_KEYS.get(field)
        recorded = yaml_key is not None and yaml_key in cfg
        override = overrides.get(field)

        if override is not None:
            if not math.isfinite(override):
                raise DynamicsProvenanceError(f"--{field.replace('_', '-')} must be finite")
            if recorded and not math.isclose(
                override, float(cfg[yaml_key]), rel_tol=1e-9, abs_tol=1e-12
            ):
                raise DynamicsProvenanceError(
                    f"{field}: explicit {override!r} contradicts the value recorded in "
                    f"config_resolved.yaml ({cfg[yaml_key]!r}). One of them is wrong -- "
                    "refusing to certify either as the trained dynamics."
                )
            dynamics[field] = float(override)
            provenance[field] = "explicit"
        elif recorded:
            dynamics[field] = float(cfg[yaml_key])
            provenance[field] = "config_resolved.yaml"
        elif field in stale_present:
            raise DynamicsProvenanceError(
                f"{field}: config_resolved.yaml carries the superseded key "
                f"'{stale_present[field]}', which today's parser ignores -- this run's "
                f"value is definitely NOT the current default "
                f"({getattr(env_config, field)!r}). Pass "
                f"--{field.replace('_', '-')} with the value that run actually trained "
                "under."
            )
        elif assume_current_defaults:
            dynamics[field] = float(getattr(env_config, field))
            provenance[field] = "assumed-default"
        else:
            unresolved.append(field)

    if unresolved:
        flags = " ".join(f"--{f.replace('_', '-')} <value>" for f in unresolved)
        raise DynamicsProvenanceError(
            "config_resolved.yaml does not record "
            + ", ".join(unresolved)
            + ", so their trained values cannot be read from this run. Present-day "
            "CrowdEnvConfig defaults would be "
            + ", ".join(f"{f}={getattr(env_config, f)!r}" for f in unresolved)
            + " -- correct only if those defaults have not changed since the run. "
            f"Either state them ({flags}) or waive the check with "
            "--assume-current-defaults, which stamps them 'assumed-default' in "
            "the artefact's provenance."
        )

    return dynamics, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="run dir with config_resolved.yaml")
    ap.add_argument("checkpoint", help="checkpoint filename inside results_dir")
    ap.add_argument("--output", type=Path, required=True, help="path for the new .onnx")
    ap.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="existing .onnx that must produce identical outputs (max |diff| <= 1e-5)",
    )
    for field in DYNAMICS_FIELDS:
        ap.add_argument(
            f"--{field.replace('_', '-')}",
            type=float,
            default=None,
            help=(
                f"the {field} this run trained under. Required unless "
                "config_resolved.yaml records it or --assume-current-defaults is set."
            ),
        )
    ap.add_argument(
        "--assume-current-defaults",
        action="store_true",
        help=(
            "accept present-day CrowdEnvConfig defaults for dynamics fields the YAML "
            "does not record, stamping them 'assumed-default' in provenance"
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.results_dir / "config_resolved.yaml")
    env_config = build_env_config(cfg)
    net_config = build_net_config(cfg, env_config)

    try:
        dynamics, dynamics_provenance = resolve_dynamics(
            cfg,
            env_config,
            {field: getattr(args, field) for field in DYNAMICS_FIELDS},
            assume_current_defaults=args.assume_current_defaults,
        )
    except DynamicsProvenanceError as exc:
        raise SystemExit(f"cannot establish the trained dynamics: {exc}") from exc

    print("dynamics (field = value [origin]):")
    for field in DYNAMICS_FIELDS:
        print(f"  {field:<24} = {dynamics[field]!r:<10} [{dynamics_provenance[field]}]")
    assumed = [f for f, origin in dynamics_provenance.items() if origin == "assumed-default"]
    if assumed:
        print(
            f"  WARNING: {', '.join(assumed)} assumed from present-day defaults, not read "
            "from the run. The artefact will certify them as trained dynamics."
        )

    ckpt = torch.load(args.results_dir / args.checkpoint, map_location="cpu", weights_only=False)
    actor_critic = ActorCritic(net_config)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()

    normalizer = None
    if "obs_normalizer" in ckpt:
        normalizer = RunningNormalizer(shape=(net_config.obs_dim,))
        normalizer.load_state_dict(ckpt["obs_normalizer"])

    out = export_onnx(
        actor_critic.actor,
        normalizer,
        args.output,
        obs_config=env_config.obs,
        action_config=env_config.action,
        dynamics=dynamics,
        provenance={
            "run": args.results_dir.name,
            "checkpoint": args.checkpoint,
            "rollout": int(ckpt.get("rollout_count", -1)),
            "git_rev": _git_rev(),
            "source": "scripts/reexport_onnx.py",
            # Per-field origin of the certified dynamics block, so a reader can
            # tell a value read from the run from one asserted at re-export.
            "dynamics_provenance": dynamics_provenance,
        },
    )
    print(f"exported: {out} ({out.stat().st_size / 1024:.1f} KB, obs_dim={net_config.obs_dim})")

    if not verify_onnx(out, actor_critic.actor, normalizer):
        raise SystemExit("ONNX output does not match the reconstructed torch model")
    print("torch parity: OK")

    if args.verify_against is not None:
        import onnxruntime as ort

        obs = np.random.default_rng(0).standard_normal((256, net_config.obs_dim))
        obs = obs.astype(np.float32)
        outputs = []
        for path in (out, args.verify_against):
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            outputs.append(session.run(None, {"observations": obs})[0])
        diff = float(np.abs(outputs[0] - outputs[1]).max())
        print(f"max |diff| vs {args.verify_against.name}: {diff:.3e}")
        if diff > 1e-5:
            raise SystemExit("re-export does not reproduce the reference artefact's outputs")

    # Prove the deployment side self-configures from the artefact alone.
    from crowdrl_jupedsim.policy import OnnxPolicy, resolve_configs

    policy = OnnxPolicy(out)
    resolved_obs, resolved_action = resolve_configs(policy)
    assert resolved_obs == env_config.obs and resolved_action == env_config.action
    prov = policy.metadata.provenance
    print(f"self-configures: obs_dim={resolved_obs.obs_dim}, provenance={prov}")


if __name__ == "__main__":
    main()

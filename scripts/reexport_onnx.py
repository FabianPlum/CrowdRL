"""Re-export a trained checkpoint as a self-describing ONNX artefact.

Rebuilds the actor + frozen observation normalizer from a results dir's
``checkpoint_rollout_*.pt`` and its ``config_resolved.yaml`` (the config
contract), then exports through ``crowdrl_train.export.export_onnx`` so the
resolved obs/action configs and provenance land in the file's
``metadata_props`` (issue #7). The result deploys in crowdrl-jupedsim with
zero hand-supplied configuration.

Typical use -- upgrade a legacy artefact in place, proving equivalence:

    uv run python scripts/reexport_onnx.py results_exp_.../ \
        checkpoint_rollout_0400.pt --output example_model/policy_r0400.onnx \
        --verify-against results_exp_.../policy_r0400.onnx
"""

from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    cfg = load_config(args.results_dir / "config_resolved.yaml")
    env_config = build_env_config(cfg)
    net_config = build_net_config(cfg, env_config)

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
        provenance={
            "run": args.results_dir.name,
            "checkpoint": args.checkpoint,
            "rollout": int(ckpt.get("rollout_count", -1)),
            "git_rev": _git_rev(),
            "source": "scripts/reexport_onnx.py",
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

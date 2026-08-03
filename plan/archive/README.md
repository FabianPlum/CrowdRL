# plan/archive -- point-in-time records

Dated records of sessions, sweeps and branch reviews. **Nothing in this folder is
maintained.** Each file is a snapshot of what was true and believed on its date; several
recommend configurations the project has since reversed.

For current state, read `plan/CrowdRL_Project_Plan_v10.md`. For how the code behaves today,
read `docs/agent_pipeline.md` and `docs/environment_mechanics.md`.

These files are kept because each holds evidence recorded nowhere else -- measured results,
root-cause analyses, and negative findings that are expensive to re-derive and easy to
re-attempt by accident.

| File | Date | Why it is kept |
|---|---|---|
| `2026-04-11_stuck_agent_tuning_summary.md` | 2026-04-11 | The noise-floor protocol (\|dGR\| > 0.055, \|dR\| > 1.43) used as the significance standard by later sweeps, and the per-tier Wilson-CI table. **Its headline recommendation was later reversed** -- see the banner |
| `2026-06-18_agent_dynamics_refactor_branch_summary.md` | 2026-06-18 | The 19-row **avenue map** (every avenue tried, with a verdict) and the explicit "do not re-try without new evidence" list. The best index of what has already been attempted |
| `agent_dynamics_refactor_merge_review.md` | 2026-06-19 | The r355/r360 **normaliser count-overflow root cause**, the fix-vs-tuning-vs-feature taxonomy with commit hashes, and the 17-row experiment lineage for the untracked `exp_*` configs |
| `handover_2026-07-20.md` | 2026-07-20 | The baked obs-normaliser statistics table for the r0400 training distribution, and the per-agent callback cost measurement |
| `handover_2026-07-29.md` | 2026-07-29 | Intermediate contact-physics measurements (min pairwise spacing 0.044 m -> 0.289 m) and the corner-clearance root-cause diagnosis. Its design rationale has been rehomed into the plan |
| `handover_2026-07-30.md` | 2026-07-30 | The Windows MSVC/CMake build incantation for JuPedSim 2.0 (and why `-vcvars_ver=14.38` matters), the `USE_LIBUV=0` DDP-test fix, and the 320-vs-322 byte-parity length asymmetry |

## Convention

A record is archived rather than deleted when it holds measurements, root causes or
negative results that are not derivable from the code or the git history. A record is
deleted when everything in it is either wrong or restated in full elsewhere -- as happened
on 2026-07-30 to `final_review_2026-07-29.md`, `pytorch_env_migration.md`,
`speedup_and_parallelisation.md` and the v5/v8 plan drafts.

Any reference in this folder to a file that no longer exists is a dangling pointer left
by that earlier pass; it is not a sign the file was misplaced.

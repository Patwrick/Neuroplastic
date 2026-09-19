> **Canonical workspace note — 2026-09-19:** The active project is now
> `C:\Users\Patrick\Documents\GitHub\Neuroplastic` on `main`.
> `research/v0.4.1` preserves completed M0–M2 work at `1cccb5f`;
> `legacy/pre-v0.4.1` preserves legacy source at `f0bd287`.
> The separate worktree was retired after all 88 M0–M2 evidence files were
> verified in this project's `outputs/v0.4.1/m0-m2-20260919/`.
> The historical record below is unchanged and describes its original execution
> time. Current README/AGENTS guidance supersedes its workspace/preparation
> instructions. This correction does not start M2A or M3.
> See [WORKSPACE_CORRECTION_REPORT.md](WORKSPACE_CORRECTION_REPORT.md) for current
> workspace status and verification results.

# CSGN v0.4.1 M0 report-back

## 1. Identity and status
Completed local_math milestone, implementation commit `c049f23962282fcc348934e2e23b94743752324b`, clean execution. Identity hashes and authority are in REPORT_BACK.md and M0/run_manifest.json.

## 2. What actually changed
Environment isolated, full reference checks passed, 47-file subset verified; original ZIP rehashed without unpacking. No legacy component reused; no released file altered.

## 3. Commands actually executed
`.\.venv-v041\Scripts\python.exe -B Docs/research/csgn_v0_4_1/verification/check_math_v0_4_1.py --output outputs/v0.4.1/m0-m2-20260919/reference_final.json` — exit 0, 3.542 seconds; log `reference_final.log`.

## 4. Tests
24 groups / 19438 local cases, passed; final zero failures/skips. Earlier bytecode inventory failures and graph fixture failure remain documented in REPORT_BACK.md.

## 5. Results, controls and resources
CPU only, isolated Python 3.13.5. Exact dependencies, commands, diagnostics and resource limits are retained in the aggregate report and environment JSON. No learned comparison.

## 6. Diagnostics
Finite difference and parity boundaries are explicit; no GPU claim. Read the aggregate report's negative findings and initial failures.

## 7. Deviations and open specification questions
See D01–D08 in REPORT_BACK.md. No confirmed central mathematical contradiction; no source equations changed.

## 8. Claim assessment
H1/H2/H3 untested. Local checks do not validate the architecture.

## 9. Next bounded action
The next authorized slice was M2; it executed and is reported in REPORT_BACK.md. No automatic M3 run.

## 10. Artifact inventory
M0/run_manifest.json, M0/run_result.json, final test XML/reference JSON, logs and artifact_inventory.json. All artifacts remained local.

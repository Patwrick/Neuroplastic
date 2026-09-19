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

# CSGN v0.4.1 M0 repository assessment

Inspected 2026-09-19 for the fresh implementation request.

- Prepared workspace: `C:/Users/Patrick/Documents/GitHub/Neuroplastic-v0.4.1`.
- Remote: `https://github.com/Patwrick/Neuroplastic.git`; branch `research/v0.4.1`.
- Initial commit: `38d3744e587153f23b8ad88ebff435626f96c9b2`; initially clean.
- Task host cwd initially pointed at the preserved `Neuroplastic` checkout on
  `rewiring` / `6fa1c850f067de36db0c25a732873aa4d72ff181`. Only its status,
  CLEAN_START_REPORT and root README were read to identify this mismatch. No
  legacy algorithm, archive payload, checkpoint, or result was inspected.
  Its five tracked modifications and untracked user files were left untouched.
- Every implementation command uses the prepared worktree explicitly. No reset,
  checkout switch, merge, push, upload, archive unpack, or legacy-default reuse.

Authority is the current user request, host instructions, root AGENTS.md, and
the paper plus handoff at the verified paths recorded in CLEAN_START_REPORT.md.
The root preparation-only instruction describes the prior task; the user's
explicit fresh M0–M2 implementation request supersedes that restriction.
The release AGENTS.md is scoped to Docs/research/csgn_v0_4_1; its contracts are
also explicitly invoked by this request and root AGENTS.md. No nested source
or test AGENTS files exist. Checked ancestor instruction locations were absent;
`C:/Users/Patrick/.codex/AGENTS.md` is empty. No global configuration was changed.

Read root README, CLEAN_START_REPORT, root/package AGENTS, WORKING_SUBSET,
handoff README, CODEX_HANDOFF, Markdown paper, IMPLEMENTATION_CONTRACT,
EXPERIMENT_PROTOCOL, CONTRACT_TEST_MAP, REFERENCE_PROFILE_DETAILS, MILESTONES,
REPORT_BACK_GUIDE, report templates, schemas, CPU config, and reference math.
The Markdown paper is the mathematical source; configs are declared choices.

Verified identities (SHA-256):

| Artifact | Digest |
|---|---|
| Intact original handoff ZIP (independently rehashed, not unpacked) | 15152533d79407f84dd95778e255624fb4c07079f0814772d3819c02b32ccae2 |
| Original release MANIFEST.sha256.json | b9546e2d3da8417d175a9174f10870c3e7cc8213a97f5f9992b89b75f9c9dc15 |
| WORKING_SUBSET_MANIFEST.sha256.json | 720329b2b3995ec43dc7f0e4d4792bd0096095f007ed582373c45f6b1a64eeb2 |
| Markdown paper | abe9fe9173bfd6fc96c9194d690d88ed795f224fa2fc3c756b0aeca167cf5f36 |

All 47 unchanged selected release files passed the subset verifier. The subset
intentionally omits three historical archive entries; original completeness is
not claimed or tested here. Full release ZIP remains outside the active tree.

Environment: prepared isolated `.venv-v041`, Python 3.13.5, no system site
packages, imported namespace resolves to this src tree, legacy `csgn` absent.
Installed NumPy 2.3.5, SciPy 1.17.0, torch 2.10.0, jsonschema 4.26.0 locally.
PyTorch build has no CUDA and reports CPU only. Hardware inventory independently
reports Ryzen 9 9950X3D (16 cores/32 logical), 100516372480 physical-memory bytes,
RTX 5090 (32607 MiB, driver 616.92). No CUDA stack or driver change was made.
CPU execution is sufficient and is the selected smoke device.

Reference rerun: all 24 groups / 19438 cases passed in 4.6343 seconds internally,
5.3926 seconds command wall time. New results/logs are under
`outputs/v0.4.1/m0-m2-20260919/`; original verification files are unchanged.
The initial rerun created two Python bytecode files in the release directory;
strict integrity checks correctly rejected those unexpected files. Both were
preserved in the evidence folder and removed individually from the release
directory. Subsequent commands use `-B`; the verifier then passed again.

Implementation namespace: src/neuroplastic_v041; tests: tests/v041. No legacy
component was reused. Dependency/setup downloads were limited to required pinned
libraries; experiments do not download data or use cloud services. Total executed
training/smoke runtime cap is 15 minutes; each smoke job is capped at 180 seconds.
M3 training, large sweeps and advanced extensions require a separate continuation.

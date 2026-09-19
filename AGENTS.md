# CSGN v0.4.1 workspace instructions

Canonical repository: `Patwrick/Neuroplastic`.
Canonical and only active project folder:
`C:\Users\Patrick\Documents\GitHub\Neuroplastic`.
`main` contains the revised project; `legacy/pre-v0.4.1` preserves legacy source
at `f0bd287`, and `research/v0.4.1` preserves the completed M0–M2 work at `1cccb5f`.
Historical reports retain their original execution paths and observations.

Read `/README.md`, then `/Docs/research/csgn_v0_4_1/WORKING_SUBSET.md` and the
handoff `/Docs/research/csgn_v0_4_1/README.md`. Read `handoff/CODEX_HANDOFF.md`,
`paper/csgn_paper_v0_4_1.md`, `handoff/IMPLEMENTATION_CONTRACT.md`,
`handoff/EXPERIMENT_PROTOCOL.md`, `handoff/CONTRACT_TEST_MAP.md`, and
`handoff/REFERENCE_PROFILE_DETAILS.md` within that package before implementation.
The paper and implementation contract define the model.

- M0–M2 is implemented. The current task is workspace correction and verification;
  do not start M2A, M3, learned training, or expanded experiments.
- Use GitHub Desktop for repository operations. Do not use terminal Git, GitHub
  CLI, or manipulate Git metadata. Preserve the original `.git` directory.
  Do not create another clone, nested repository, or linked worktree. No history
  rewriting, force-pushing, remote changes, or publication of data is authorized.
- The user's workspace-correction request authorizes merging the revised branch
  into `main` and removing obsolete active files after legacy preservation is
  confirmed. Earlier preparation-only and merge-prohibition instructions are
  superseded. Preserve unrelated repository settings and all revised evidence.
- Use `src/neuroplastic_v041`, `tests/v041`, and the canonical `.venv-v041`.
  Never import legacy `csgn`, load old checkpoints, restore old defaults, or impose
  historical backward compatibility. Review legacy algorithms or archives only
  for a concrete unresolved issue, recording its scope, rationale, and findings.
- Validate `WORKING_SUBSET_MANIFEST.sha256.json` with the root verification script.
  Historical archive entries are explicitly omitted; never report the subset as
  a complete original release. Preserve shipped paper, reference, and manifest
  bytes. Do not change the paper to fit implementation results.
- Report specification discrepancies and stop the affected work. Preserve
  continuous outer gradients through the reference write path; detached paths
  are explicit ablations. Keep causal feedback boundaries and independent
  per-lifetime state. Reference checks and smoke tests do not establish learned
  capability, global convergence, or architectural validity.
- Preserve all seeds and negative, failed, aborted, inconclusive, and positive
  results. New evidence uses `outputs/v0.4.1/<unique-run-id>/` and the handoff
  schemas/templates. The original 88-file M0–M2 evidence directory is immutable;
  correction evidence belongs in `outputs/v0.4.1/workspace-correction-20260919/`.
  Ignored outputs are not disposable. Do not rerun historical evidence-writing
  helpers against their original output directories.
- Execute and report tests honestly. Three existing `test_capture_` provenance
  tests invoke command-line Git; deselect them explicitly while the Desktop-only
  restriction applies. The smoke CLI also invokes Git for provenance. Never hide
  skipped checks, fabricate results, or relax tolerances just to make tests pass.
  Record installed versions with `importlib.metadata`; `pip freeze` can invoke
  Git internally for this editable installation and is not allowed here.
- No large sweeps, paid/cloud compute, Minecraft, rewiring, INT4/offloading, online
  shared-encoder changes, or asynchronous sleep. Preserve existing runtime limits.
- Leave credentials and `.env` files untouched. Do not commit environments, raw
  experiment outputs, or secrets. Do not create more archive folders or recursively
  unpack historical archives. Existing preservation records remain external.
- Package instructions remain scoped to their package. This root guidance carries
  applicable research and security boundaries into source, tests, and configs.

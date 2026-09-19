# CSGN v0.4.1

The canonical project folder is
`C:\Users\Patrick\Documents\GitHub\Neuroplastic` in the single
`Patwrick/Neuroplastic` repository. `main` contains the revised v0.4.1 project.
Use GitHub Desktop for repository operations; do not use terminal Git, GitHub CLI,
or direct Git metadata edits. Do not create another clone or linked worktree.

See [WORKSPACE_CORRECTION_REPORT.md](WORKSPACE_CORRECTION_REPORT.md) for current
workspace status, preservation checks, and actual verification results.

M0–M2 implemented the bounded recurrent graph, continuous functional write path,
causal T1 task, synchronous slow-state fitting, and simple delta-memory baseline.
The completed smoke experiment is a correctness exercise. Its weak graph cold
recall and harmful writes remain reported; it does not validate the architecture.
The current task is workspace correction and verification. Do not start M2A or M3.

## Research authority and source

Read [AGENTS.md](AGENTS.md), the verified
[working subset](Docs/research/csgn_v0_4_1/WORKING_SUBSET.md), and the
[handoff README](Docs/research/csgn_v0_4_1/README.md). Then read
[CODEX_HANDOFF.md](Docs/research/csgn_v0_4_1/handoff/CODEX_HANDOFF.md), the
[v0.4.1 paper](Docs/research/csgn_v0_4_1/paper/csgn_paper_v0_4_1.md), and the
[implementation contract](Docs/research/csgn_v0_4_1/handoff/IMPLEMENTATION_CONTRACT.md).
The paper defines the mathematics, the contract operationalizes it, and configs
are declared engineering choices. Report discrepancies; do not import old designs.

Implementation lives in [src/neuroplastic_v041](src/neuroplastic_v041), tests in
[tests/v041](tests/v041), and the bounded configuration in
[configs/m2_smoke.json](configs/m2_smoke.json). Shipped handoff bytes and manifests
remain unchanged. The 47-file subset is not the complete original release.
The immutable original `csgn_v0_4_1_codex_handoff.zip` remains at the repository
root as a local, ignored release copy. The authoritative active files are the
verified 47-file subset under `Docs/research/csgn_v0_4_1/`.

## Local environment and verification

Run commands from the canonical folder. Use a fresh local `.venv-v041` environment;
do not activate the legacy `.venv` or add historical code to `PYTHONPATH`.

```powershell
py -3.13 -m venv .venv-v041
.\.venv-v041\Scripts\python.exe -m pip install -r Docs/research/csgn_v0_4_1/requirements-reference.txt
.\.venv-v041\Scripts\python.exe -m pip install -e '.[test]'
.\.venv-v041\Scripts\python.exe -I -B scripts/verify_handoff_subset.py
.\.venv-v041\Scripts\python.exe -B -m pytest -q -k 'not test_capture_'
```

The last command runs the existing suite while deselecting three provenance
integration tests that invoke command-line Git to create temporary repositories.
Report those exclusions explicitly. The original M0–M2 evidence records 163 tests
passing before the Desktop-only restriction; correction results must report their
own actual executed counts. The smoke CLI also invokes command-line Git for
provenance and is not part of this workspace-correction run.

## Evidence and preservation

The original M0–M2 evidence is retained unchanged in
`outputs/v0.4.1/m0-m2-20260919/`: all 88 files were copied into this canonical
project and verified by size and SHA-256. Correction evidence belongs in
`outputs/v0.4.1/workspace-correction-20260919/`; new runs always need a distinct
directory. Never overwrite prior results, including failed or negative findings.
Outputs are local and ignored by Git.

The local `research/v0.4.1` branch preserves the completed implementation and
reports at `1cccb5f`. The local `legacy/pre-v0.4.1` branch preserves the old
implementation and its outstanding source changes at `f0bd287`. Legacy material
is for targeted inspection when a concrete unresolved issue requires it; record
the reason and findings. Existing external preservation records are indexed in
[ARCHIVE_INDEX.md](ARCHIVE_INDEX.md). No new archive workflow is needed.

[REPORT_BACK.md](REPORT_BACK.md), [M0_REPORT.md](M0_REPORT.md),
[M1_REPORT.md](M1_REPORT.md), and [M2_DECISIONS.md](M2_DECISIONS.md) retain the
original results and limitations. Historical path and branch statements in those
reports describe their execution time; their location notes identify this folder
as the current project.

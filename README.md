# CSGN v0.4.1 implementation workspace

Status: preparation skeleton; no integrated model, trainer, or learned results.

This is the **root repository README**. Start with the verified working subset
at [Docs/research/csgn_v0_4_1/WORKING_SUBSET.md](Docs/research/csgn_v0_4_1/WORKING_SUBSET.md).
The **handoff README** is [Docs/research/csgn_v0_4_1/README.md](Docs/research/csgn_v0_4_1/README.md).
Read its handoff/CODEX_HANDOFF.md, paper/csgn_paper_v0_4_1.md, and
handoff/IMPLEMENTATION_CONTRACT.md in that order. The paper defines mathematics;
the handoff operationalizes it; configs are candidate engineering settings.
Report discrepancies instead of importing older designs.

Verify the subset without importing research code:

```powershell
python -I scripts/verify_handoff_subset.py
```

For the future implementation task, create a fresh environment in this worktree:

```powershell
py -3.13 -m venv .venv-v041
.\.venv-v041\Scripts\python.exe -m pip install -r Docs/research/csgn_v0_4_1/requirements-reference.txt
.\.venv-v041\Scripts\python.exe -m pip install -e '.[test]'
```

Do not activate the legacy checkout's environment or set PYTHONPATH to it.
Source belongs in src/neuroplastic_v041; pytest is scoped to tests/v041.
No implementation tests exist yet; an empty test collection is not a pass.
New evidence belongs under outputs/v0.4.1/<unique-run-id>/ and must use the
handoff schemas. Never overwrite shipped verification results or load an old
checkpoint as a starting constraint. See ARCHIVE_INDEX.md for historical evidence.

M1/M2 and learned runs require a fresh implementation task. This preparation
contains no runnable training command. Legacy results are evidence only for their
original version, including negative, failed, aborted, and inconclusive outcomes.

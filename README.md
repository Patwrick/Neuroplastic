# CSGN v0.4.1 implementation workspace

Status: M0–M2 correctness implementation: bounded recurrent graph, functional
projected writes, causal T1 task, synchronous slow fitting and delta baseline.
Fixed-theta smoke execution does not validate the architecture or H1/H2/H3.

This is the **root repository README**. Start with the verified working subset
at [Docs/research/csgn_v0_4_1/WORKING_SUBSET.md](Docs/research/csgn_v0_4_1/WORKING_SUBSET.md).
The **handoff README** is [Docs/research/csgn_v0_4_1/README.md](Docs/research/csgn_v0_4_1/README.md).
Read its handoff/CODEX_HANDOFF.md, paper/csgn_paper_v0_4_1.md, and
handoff/IMPLEMENTATION_CONTRACT.md in that order. The paper defines mathematics;
the handoff operationalizes it; configs are candidate engineering settings.
Report discrepancies instead of importing older designs.

Verify the subset without importing research code:

```powershell
python -I -B scripts/verify_handoff_subset.py
```

Use an isolated environment in this worktree:

```powershell
py -3.13 -m venv .venv-v041
.\.venv-v041\Scripts\python.exe -m pip install -r Docs/research/csgn_v0_4_1/requirements-reference.txt
.\.venv-v041\Scripts\python.exe -m pip install -e '.[test]'
```

Do not activate the legacy checkout's environment or set PYTHONPATH to it.
Source belongs in src/neuroplastic_v041; pytest is scoped to tests/v041.
Implementation tests cover endpoint math, continuous outer gradients, routing,
causal feedback, shadow rejection, state isolation and persistent cold recall.
New evidence belongs under outputs/v0.4.1/<unique-run-id>/ and must use the
handoff schemas. Never overwrite shipped verification results or load an old
checkpoint as a starting constraint. See ARCHIVE_INDEX.md for historical evidence.

The fresh implementation request authorizes M0–M2. Run the contract suite:

```powershell
.\.venv-v041\Scripts\python.exe -B -m pytest -q
```

Run the bounded fixed-theta graph/delta lifecycle with a new output directory:

```powershell
.\.venv-v041\Scripts\python.exe -B -m neuroplastic_v041.experiments.smoke --output outputs/v0.4.1/my-smoke
```

The command caps execution at 180 seconds, retains failed/partial results, and
refuses to overwrite existing runs. It trains only a shadow slow-state candidate
during four sleep steps; it does not train shared representations. Exact source,
dirty diff, configs, manifests, per-lifetime metrics and diagnostics accompany
each run. See REPO_ASSESSMENT.md, M2_DECISIONS.md and REPORT_BACK.md for observed
results, instruction precedence, limitations and the next bounded action.
M3 training and later stages require separate review. Legacy results remain
evidence only for their original version, including negative or failed outcomes.

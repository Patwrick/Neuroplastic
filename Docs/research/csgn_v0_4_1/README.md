# CSGN v0.4.1 — research-to-implementation handoff

**Release:** 16 September 2026. **Target repository:** `Patwrick/Neuroplastic`.
**Status:** revised research specification plus executable local references; **not** a complete model or a trained checkpoint.

## Start here

Give Codex this entire extracted folder in its working environment, not only the PDF. In the Neuroplastic checkout, a suitable location is `Docs/research/csgn_v0_4_1/`. The location is a suggestion; do not overwrite an existing folder or existing agent instructions without reviewing them. Use [START_CODEX_PROMPT.md](START_CODEX_PROMPT.md) as the first task. Codex must verify the actual checkout and package before modifying code.

The requested “restart” is a new implementation track, **not permission to delete legacy research, reset history, erase uncommitted work, or discard data**. No repository changes were made in preparing this package.

## Read in this order

1. [Codex handoff](handoff/CODEX_HANDOFF.md): goal, scope, first work slice and definition of done.
2. [v0.4.1 paper](paper/csgn_paper_v0_4_1.md), also [PDF](paper/csgn_paper_v0_4_1.pdf): normative mathematics and claim limits.
3. [Implementation contract](handoff/IMPLEMENTATION_CONTRACT.md): state, interfaces, update order and differentiation.
4. [Experiment protocol](handoff/EXPERIMENT_PROTOCOL.md): tasks, controls, baselines and diagnostics.
5. [Milestones](handoff/MILESTONES.md), [contract/test map](handoff/CONTRACT_TEST_MAP.md), and [report-back guide](handoff/REPORT_BACK_GUIDE.md).

The paper is the scientific source of truth. The contract operationalizes it; configs are candidate engineering settings. Reference tests check claims but do not redefine the paper. On a conflict, record a specification issue and stop only the affected work; do not silently choose whichever interpretation yields a passing result. Historical archives are **non-authoritative**.

## What is supplied and what Codex must build

**Supplied now:** revised manuscript, derivations, NumPy/PyTorch local reference functions, 24 executed local check groups, raw results and source hashes, bounded experiment configs, schemas, dependency list, report templates, package validator, environment probe, report-bundling helper, and historical research archives.

**Not supplied:** an integrated CSGN trainer, environment/task implementation, learned encoder/router, sparse kernels, complete benchmark runner, model checkpoint, or evidence of H1/H2/H3. Building and testing those is Codex's assignment. A test of an ideal matrix memory does not count as a trained graph experiment.

## Immediate runnable commands

Use an isolated Python environment. The reference package was checked with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and PyTorch 2.10.0+cpu. See the shipped results for the precise interpreter version. The dependency pins identify releases; platform wheels/BLAS can differ. Do not replace an existing project environment blindly.

```bash
python -m pip install -r requirements-reference.txt
python scripts/validate_package.py
python scripts/environment_probe.py --output environment_probe.local.json
python verification/check_math_v0_4_1.py --output verification/rerun.local.json
```

These commands test local contracts only. Future training commands are deliberately not advertised as runnable until Codex implements them. The original release result is immutable; reruns use a new filename. The package manifest lists release files only, so extra local rerun files do not change its integrity result.

## Essential changes from v0.4

- Exact four-endpoint feasibility permits safe inward corrections at saturated slow weights.
- Inner synaptic partial derivatives hold features fixed, while outer meta-training retains continuous paths through the write.
- Required diagnostics distinguish capacity, conditioning, step restrictions, and downstream usefulness.
- Stable addressing is an ablation, not an assumed improvement.
- Slow-memory transplantation, selective erasure, negative controls and held-out rule transfer test why learning works.
- Simple delta memory is a primary competitor; confidence is reported across independent trained seeds.
- Validation has separately labeled empirical, independent research, and optional formal levels.
- The first model keeps the full compute–write–sleep–clear–recall loop but defers rewiring, INT4, offloading, online shared-encoder changes, asynchronous sleep, and learned internal teachers.

## Evidence and handoff loop

Release verification passed **24 groups / 19,438 local cases**. Inspect [raw results](verification/results_v0_4_1.json), not just this summary. No integrated performance claims are made.

After each milestone, Codex must fill [REPORT_BACK.md](templates/REPORT_BACK.md), save machine-readable manifests/results, and identify failed or unexecuted checks. Share that report plus logs/metrics and any specification issues back for research review. Model-generated prose alone is not enough to revise a scientific claim.

[CHANGELOG](paper/CHANGELOG_v0_4_1.md) · [source provenance](provenance/sources.json) · [release hashes](MANIFEST.sha256.json)

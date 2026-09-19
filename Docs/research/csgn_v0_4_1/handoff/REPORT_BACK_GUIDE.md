# Report-back and research revision loop

Return evidence after every milestone so subsequent research revisions distinguish faulty code, invalid experiments, and a genuine architectural limitation.

## The minimum return bundle

Fill `templates/REPORT_BACK.md` and save as `REPORT_BACK.md` in the run or milestone directory. Include:

- Exact code commit, dirty status/diff hash, paper/package/config hashes, environment and device.
- Executed commands with exit codes, wall time, logs, test counts and explicit skipped/blocked checks.
- `run_manifest.json` and `run_result.json` using the supplied schemas; per-seed metrics rather than only aggregate prose.
- Results for relevant baselines/controls and resource charges. Missing values remain null or not_run, not zero.
- Specification issues with section/equation, minimal reproducer, observed/expected behavior, and proposed options.
- Deviations made before a run, why they were necessary, and how they affect the claims.
- Next recommended bounded experiment and its expected information value; do not merely ask to run a much larger model.

Use `scripts/make_report_bundle.py` to collect an allowlisted local report folder. Review it before sharing; filenames alone cannot guarantee absence of confidential information. This script does not upload anything. Large checkpoints and raw private datasets stay outside the default bundle; record their hashes and retrieval location instead.

## How evidence changes the paper

1. **Implementation defect:** repair code/tests; no mathematical claim changes unless the implementation revealed an ambiguity.
2. **Specification ambiguity:** create a decision record, obtain a scoped resolution, add a regression test, then amend the paper/change log in a later release.
3. **Learning failure with valid controls:** report the negative result and diagnose reachability, gradients, target information, feature drift, interference, or capacity. Do not call a mathematical identity false because a task score is poor.
4. **Positive pilot:** state the narrow tested conditions; independent confirmation is still required.
5. **Baseline dominance:** simplify or revise the mechanism; do not weaken the comparator or drop losing seeds.
6. **Claim-changing finding:** supply raw evidence and a proposed wording/assumption change in `CLAIM_UPDATE.md`. Do not silently change the authoritative manuscript inside the implementation task.

## Requested next message to the research reviewer

“Here is Codex's v0.4.1 report-back package. Audit code/spec conformance, verify which metrics support the claims, distinguish bugs from model limitations, and propose the next bounded experiment or paper revision.”

Text summaries are welcome, but retain logs, configs, diffs and numerical results. Reproducible contrary evidence is more useful than an enthusiastic conclusion.

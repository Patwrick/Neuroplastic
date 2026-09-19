# Verification evidence

`results_v0_4_1.json` is an actual execution of `check_math_v0_4_1.py` with the source hashes recorded inside it. There are 24 check groups and 19,438 local cases. All passed the programmed tolerances. The script requires NumPy, SciPy and PyTorch; an absent dependency is a failure to execute, not an implicit pass.

The suite checks endpoint feasibility, exact interval projection, local descent, independent decay, transfer, known counterexamples, bounded-LS diagnostics, and continuous meta-gradients by finite difference. It does not contain a trained graph, a full CSGN environment, a pilot learning run, or causal memory evidence for the integrated model. The ideal matrix transplant is only a fixture illustrating the protocol.

Run from the extracted package root:

```bash
python verification/check_math_v0_4_1.py --output verification/rerun.local.json
```

Keep the release result immutable. Do not overwrite it with a partial or skipped run. Source changes require a new result and must not be presented as an unchanged release. Finite-difference tests avoid branch boundaries; no test proves exact differentiation through discrete top-k switches.

The contract map lists the important implementation tests Codex must still write. Config and schema checks are performed by `scripts/validate_package.py`; these establish file/schema consistency, not model validity. PDF visual inspection and build warnings are separately recorded in `release_build.json`.

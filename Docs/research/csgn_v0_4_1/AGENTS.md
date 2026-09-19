# CSGN v0.4.1 package guidance

This guidance applies to the extracted package. Respect all applicable checkout instructions; do not overwrite them with this file.

- Read README and handoff/CODEX_HANDOFF.md first. The paper defines mathematics; configs are declared starting choices.
- Preserve shipped paper, verification results and archive hashes. Put new results in a new run directory.
- “Restart” is non-destructive. No reset --hard, clean -fd, history rewrite, deleted datasets, force-push, visibility changes, or unapproved cloud spending.
- First build M0–M2. Do not stop at a plan when implementation and CPU tests are possible.
- Preserve continuous outer gradients during meta-training. Snapshot-fixed is not globally detached.
- Keep causal feedback boundaries and independent per-agent memory; no future-label leakage or cross-batch state sharing.
- No fabricated results, hidden skipped tests, relaxed tolerances for convenience, or claims of global convergence from local checks.
- Record deviations before interpreting results. Negative results are useful evidence.
- Use templates/REPORT_BACK.md and the result schemas. Do not auto-edit the research paper to fit the implementation.
- Copy short, relevant project guidance into the actual implementation scope only after reviewing existing instructions. Do not assume a deeply nested package AGENTS file applies to source files elsewhere in the repo.

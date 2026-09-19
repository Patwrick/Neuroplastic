# Starting prompt for Codex

Let's restart the **implementation track** of `Patwrick/Neuroplastic` using the revised **CSGN v0.4.1** research paper and this complete handoff package.

Read `README.md`, `handoff/CODEX_HANDOFF.md`, `paper/csgn_paper_v0_4_1.md`, `handoff/IMPLEMENTATION_CONTRACT.md`, `handoff/EXPERIMENT_PROTOCOL.md`, `handoff/MILESTONES.md`, and all applicable repository `AGENTS.md` instructions. Validate the release manifest and inspect the real checkout before making assumptions.

“Restart” means a clean, separately identifiable v0.4.1 implementation, not deletion of existing code, history, papers, runs, data, or uncommitted changes. Use the supplied branch/worktree if the environment has one; otherwise make a non-destructive implementation branch. Do not merge, force-push, change repository visibility, or publish raw data. Keep old experiments reproducible where practical, but do not let old equations silently define the new model.

Start with **M0–M2**: inspect and document the repository/environment, run the supplied local checks, implement the v0.4.1 mathematical contracts in independently testable PyTorch, and build the smallest complete graph–write–sleep–clear–recall smoke test alongside a simple delta-memory baseline. Follow the first-task runtime limits. Produce real code and executed tests, not just a plan. A smoke run is not scientific confirmation.

The full goal is learned representations and useful plastic writes, durable slow recall, causal memory interventions, held-out rule transfer, and fair comparison with simpler memory. Do not assume the graph must win. Keep exact short-episode outer gradients through continuous write paths; compare a clearly named detached-path ablation. Diagnose snapshot reachability before tuning away a failure.

Do not launch paid/cloud compute, a large sweep, Minecraft integration, INT4/offloading, online encoder changes, rewiring, or asynchronous sleep yet. If GPU access is absent, complete CPU checks and provide the exact bounded next command. Never invent metrics, relax a test to get a pass, or treat random algebra checks as trained-agent evidence.

When this slice is finished, return `REPORT_BACK.md`, run manifests/results, executed commands and logs, current commit/dirty state, changed files, unresolved specification questions, and the next recommended experiment. Use the supplied report templates and schemas. Continue with safe, resolvable work; ask only about genuine blockers or choices that materially change the scientific specification.

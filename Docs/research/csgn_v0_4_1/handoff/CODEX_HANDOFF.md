# Codex handoff — v0.4.1

## Mission

Implement and test a small complete CSGN lifecycle in `Patwrick/Neuroplastic`. Establish whether objective-based writes to a sparse recurrent graph produce useful adaptation and durable slow recall beyond strong simpler baselines. The job is to produce discriminating evidence, not to make CSGN appear successful.

**Authoritative manuscript:** `paper/csgn_paper_v0_4_1.md`. A PDF is supplied for human review; use Markdown and the contract map for exact equations. Prior v0.3/v0.4 papers and old code are historical, not normative defaults.

## Observed repository context

A read-only inspection during packaging found default branch `main`, current commit `281746c9ae962634df04ec34b3ba212af11b3cfd`, and a README describing v0.3-era prototypes under `csgn/model/`, `csgn/models/`, `csgn/agents/`, environment adapters and `scripts/`. The README separates the neural cue model from the old graph prototype. Only metadata/README were re-inspected for this handoff; this is not a full current code audit. The live checkout may have advanced. Verify it and its instructions; **never reset to this observed commit just because it is listed here**.

## First task: M0–M2, then a report

1. Read instructions and source documents. Inventory the checkout, dirty files, tests, dependency manifests, device and current branches/worktrees. Save `REPO_ASSESSMENT.md`; do not spend the entire task auditing legacy implementation math.
2. Validate the package and rerun its local checks in a separate environment or compatible existing environment. Preserve the original result. Record exact failures, source hashes and versions.
3. Create a separate implementation namespace, provisionally `neuroplastic_v041/`, after inspecting repository layout. Implement endpoint intervals, functional projected writes, state/feedback records, exact small-scale routing, the bounded graph workspace, snapshot construction, task interfaces, and synchronous slow fitting. Use the reference files as test oracles, not runtime NumPy wrappers inside a PyTorch graph.
4. Provide a simple delta-memory comparator using the same feedback contract. Build a tiny complete T1 lifecycle smoke run, including old/new contexts, sleep, permanent fast-off evaluation and disabled answer retrieval.
5. Execute tests under the CPU smoke budget. Return a report that separates local math, code correctness, tiny integration execution, and actual learned performance.

A clean implementation may reuse logging, seed helpers or environment interfaces after tests confirm the contract. It must not reuse incorrect learning rules under new names. Large legacy refactors, removing Minecraft support, deleting data, and rewriting repository history are not part of this request.

## Scope and autonomy

Use the environment's provided branch/worktree. If none is designated and creating a branch is permitted, use `research/csgn-v0.4.1-restart`; record a collision-resolved name rather than overwriting an existing branch. Preserve uncommitted work. Commit reviewable changes when the host workflow permits; do not merge or force-push. Never claim the repository was updated unless a tool actually confirms the write.

CPU is sufficient for M0/M1 and a tiny M2 smoke. Detect the actual GPU and validate PyTorch compatibility rather than assuming the user's historical hardware is available in the agent sandbox. Reference pins describe a tested CPU environment, not a mandatory CUDA stack. Do not install a global CUDA toolkit or mutate system drivers. Use project-local environments.

The first task has a proposed **15-minute total runtime budget for executed training/smoke jobs**, excluding ordinary source editing, with **no external paid compute**. Use the much smaller per-job config budgets. Stop gracefully and report a bounded next command if a run will exceed budget. The later pilot budget is at most two hours on one available device and requires a separate user-approved continuation after the first report. Never run a confirmation grid merely because a config exists.

## Scientific boundaries

- Local descent applies only to the fixed-feature objective. Measure the actual future query effect separately.
- Outer meta-gradients must remain connected through continuous writes in the exact reference profile; hard router selection remains a declared branch/surrogate choice.
- Do not equate fitted snapshot loss with reachable full-network behavior.
- No labels in observation dictionaries, target-coded episode names, hidden RNG seeds passed to the model, or future feedback before prediction.
- Batch elements are independent agents/lifetimes. Slow/fast state must not be averaged across them unless explicitly modeling one shared learner with a justified aggregation objective.
- Cold recall disables both residuals and writes throughout evaluation, resets transient state, and excludes episodic answers. Task context is provided only to the extent legitimately observable.
- A delta baseline is not a weak straw man. Charge replay, sleep, validation, state storage, hyperparameter search and data movement.
- Report negative, inconclusive, failed and aborted runs. Never delete inconvenient seeds or label a failed theorem check as a harmless model result.

## Definition of done for the first slice

The checkout has an isolated implementation, unit tests matching the revised endpoint and gradient contracts, a causal T1 task, exact sparse or explicitly labeled dense-reference execution, a delta-memory baseline, and an end-to-end runnable smoke command that really executes sleep and fast-off recall. The report includes test counts, genuine metrics or `not_run`, resource use, deviations, and the next milestone. A trained advantage, GPU scale, rewiring, and a formal lifetime acceptance certificate are **not** prerequisites for this slice.

## Next after review

Proceed to M3 learned-feature pilots and M4 causal/transfer experiments. Only after those results should optional structural adaptation or systems optimization be considered. A failure to outperform the simpler memory is a reason to revise or simplify, not to change the target metric after seeing results.

## Codex documentation

Official OpenAI guidance describes project instructions in `AGENTS.md`; current documentation was checked at `https://developers.openai.com/codex/guides/agents-md` (redirecting to `https://learn.chatgpt.com/docs/agent-configuration/agents-md`) on 16 September 2026. This handoff does not depend on a specific model name, product UI, or CLI flag. Read the actual host instructions and preserve their precedence. The starting prompt explicitly names the package because a nested instruction file is not automatically a repository-wide mandate.

> **Canonical workspace note — 2026-09-19:** The active project is now
> `C:\Users\Patrick\Documents\GitHub\Neuroplastic` on `main`.
> `research/v0.4.1` preserves completed M0–M2 work at `1cccb5f`;
> `legacy/pre-v0.4.1` preserves legacy source at `f0bd287`.
> The separate worktree was retired after all 88 M0–M2 evidence files were
> verified in this project's `outputs/v0.4.1/m0-m2-20260919/`.
> The historical record below is unchanged and describes its original execution
> time. Current README/AGENTS guidance supersedes its workspace/preparation
> instructions. This correction does not start M2A or M3.
> See [WORKSPACE_CORRECTION_REPORT.md](WORKSPACE_CORRECTION_REPORT.md) for current
> workspace status and verification results.

# M0–M2 decisions made before the integration run

1. Use the shipped CPU smoke dimensions: 64 columns, 512 slots, 16 active
   senders, 4 rounds, 4 sleep Adam steps at .001, two generated lifetimes, fixed
   random theta, and CPU FP32. Local derivative/parity tests use FP64. No trained
   representation or architecture validation claim follows from these runs.
   The first smoke accidentally inspected every write while leaving the config's
   diagnostic sampling rate at 0.1. Preserve that run as an execution with this
   discrepancy. The final local configs/m2_smoke.json explicitly sets the rate to
   1.0 (still capped at 8 calls), and the runner now honors the rate with a separate
   deterministic RNG. All other scientific/resource settings remain unchanged.
2. The graph computes bounded typed messages and captures actual aggregate
   outputs. Its four-output averaging decoder is common to the write snapshot.
   Protected output-delivery topology may have writable weights. Incoming and
   updated-node budgets are hard limits; rejected route mass is not redistributed.
3. Dense B-by-E state, dense source/head computation and all-edge logits are
   explicitly charged. Gather/scatter aggregates avoid E-by-N-by-width storage.
   Region summaries are recomputed in this small reference. No sparse-speed claim.
4. WM is a zero-capacity no-write profile; no answer retrieval path exists.
   Pending queue: 32 records, 16000000 charged bytes, max delay 16, reject newest
   on overflow. Replay: 128 records, 32768 charged bytes, FIFO eviction. These
   finite byte caps fill an underspecified config detail, and count tensor bytes
   plus declared scalar metadata; Python object overhead is separately limited
   only by finite record counts and measured process memory. Demand queues are
   capped at N requests/lifetime, reject newest, and record event service latency.
5. Support predicts then reveals feedback. Queries are scorer-only cloned
   probes; live support/distractor workspace continues without probe state.
   Cold probes reset h/WM/queues once per episode, disable both residuals for all
   steps, and never call a write/retrieval helper. Eight evaluation episodes and
   eight empirical-development episodes use distinct query sampling seeds per
   lifetime; they share the same association table and are not independent seeds.
6. Sleep is full-graph recurrent replay, not frozen-design least squares.
   Shadow S alone receives four Adam steps, with theta/G frozen, both residuals
   zero and S projected to bounds. Replay sequence resets at each optimizer step.
   One fixed candidate is screened against old/new cold development queries.
   Reject preserves live state/pending records; accepted reset-boundary publication
   clears pending records with an explicit lost-credit count. Transaction ID and
   feedback cutoff are logged independently of unchanged theta/G/feature versions.
7. Delta memory uses the tensor product of bounded cue and context (128 key
   coordinates) and an 8x128 per-lifetime matrix. It receives the same feedback,
   replay records, four Adam steps and empirical screen. Its normalized delta
   writes use storage bounds without graph step restrictions. It is a substantive
   engineering baseline; feature geometry, bytes and latency are not matched.
8. Diagnostics solve the storage-only box on captured graph factors, report
   primal/dual bounds and numerical gaps, and do not feed oracle answers to the
   learner. With/without-write probes share post-decay state and query draws.
   Partial-feature detach is an explicit ablation; default writes retain continuous
   outer gradients through recurrent features, selected gates and normalization.
9. Main scope is the minimum T1 lifecycle. Post-cold supersession, T2 delay sweeps,
   transplants, erasure, T3 transfer, learned features and independent comparisons
   are not run. Delayed/reordered feedback and stale generations are unit tested.
   No formal lifetime certificate, paid compute or advanced systems extensions.
10. No confirmed mathematical contradiction was found in the central contracts.
    Ambiguous profile counts (64 senders example vs config's 16), absent byte
    caps and episode/reset interpretation are resolved explicitly above. Any
    scientific discrepancy discovered later stops only affected work; shipped
    documents are never changed to accommodate implementation behavior.

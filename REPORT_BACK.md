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

# CSGN v0.4.1 report-back

## 1. Identity and status

- Milestone/run: M0–M2 / `m0-m2-20260919`; **completed** as a bounded implementation slice.
- Evidence classes: local_math and implementation_smoke. H1/H2/H3 remain untested.
- Repository: Patwrick/Neuroplastic; prepared workspace `C:/Users/Patrick/Documents/GitHub/Neuroplastic-v0.4.1`; branch `research/v0.4.1`.
- Start commit: `38d3744e587153f23b8ad88ebff435626f96c9b2`, clean. Tested implementation commit: `c049f23962282fcc348934e2e23b94743752324b`, clean at final tests/smoke. Empty dirty patch SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. A later report-only commit records this report; use `git log -1 --format=%H -- REPORT_BACK.md` for that identity.
- Original release ZIP SHA-256: `15152533d79407f84dd95778e255624fb4c07079f0814772d3819c02b32ccae2`, independently hashed at the verified private path without unpacking.
- Original manifest SHA-256: `b9546e2d3da8417d175a9174f10870c3e7cc8213a97f5f9992b89b75f9c9dc15`.
- Working-subset manifest SHA-256: `720329b2b3995ec43dc7f0e4d4792bd0096095f007ed582373c45f6b1a64eeb2`; all 47 selected files verified, **original completeness not claimed**.
- Paper SHA-256: `abe9fe9173bfd6fc96c9194d690d88ed795f224fa2fc3c756b0aeca167cf5f36`. Final config SHA-256: `a21ac7eaa7689f71448f471e36dc9f82d51cf7c3ab9c076e361b9b4cae6cff75`. Combined task split SHA-256: `e0150bf96d97f13d32c78f939daca76aef0494f9461fc324865c5316c8d73914`; individual split IDs are in each lifetime JSON.
- Date/device: 2026-09-19, Windows 11, isolated Python 3.13.5, torch 2.10.0 CPU, NumPy 2.3.5, SciPy 1.17.0, jsonschema 4.26.0; FP32 smoke, FP64 contract/derivative fixtures. Final smoke uses one CPU thread. Detected RTX 5090 is not CUDA-accessible through this CPU wheel; no GPU execution claimed.
- Sources: current user/host instructions, root README/AGENTS/CLEAN_START_REPORT, working-subset note, handoff README/CODEX_HANDOFF, paper, implementation contract, experiment protocol, test map, profile, milestones and report guide. Root preparation-only text belongs to the previous task; the fresh user request authorizes M0–M2. No applicable nested source/test instructions or ancestor overrides were found; global AGENTS is empty. See REPO_ASSESSMENT.md.

## 2. What actually changed

Implemented functional batched endpoint feasibility and interval projection (Eq4,18–21), independent decay and exact transfer, causal bounded feedback/replay, fixed-slot topology and exact routing/null/admission, bounded recurrent workspace (Eq8–12), actual aggregate/A snapshots, T1 observation interfaces, full-graph synchronous slow fitting, persistent cold evaluation, and a context-keyed normalized delta baseline (Eq16). Shared continuous outer paths remain connected; feature detachment is an explicitly tested ablation.

Source is under `src/neuroplastic_v041`, tests under `tests/v041`, and the runnable CPU entrypoint is `neuroplastic_v041.experiments.smoke`. Pinned dependencies are declared in pyproject.toml. `scripts/run_logged.py` retains actual command exits, wall times and logs. `M2_DECISIONS.md` records finite memory budgets, dense execution, queue policy, baseline geometry and other scoped decisions. Runtime learning does not wrap NumPy references.

No legacy component was reused. The initial host cwd was the preserved `rewiring` checkout; its report/status/root README identified the correct worktree. No legacy algorithms, archives, checkpoints or old results were inspected. No targeted historical review was needed. The original five tracked modifications and untracked user files remain in the same status. No merge, push, upload, paid/cloud job, Minecraft, rewiring, INT4, offload, online theta changes or asynchronous sleep occurred.

Deferred: M3 shared-encoder training, matched-resource/replay-only/static trained comparisons, post-cold T1 supersession, T2 sweeps, T3 rule transfer, transplants, erasure, independent-seed uncertainty and formal acceptance. No undocumented command is represented as trained execution.

## 3. Commands actually executed

All table commands ran in the prepared workspace. Logs below live under `outputs/v0.4.1/m0-m2-20260919/`. Recorded durations are command wall seconds, including interpreter startup. The inner manifest timestamp is created after source capture while its elapsed timer starts before capture; use the external command log for authoritative launch/end timing. This approximately one-second provenance-timing offset does not change the numerical results. The original pinned dependency install also executed successfully (`.\.venv-v041\Scripts\python.exe -m pip install -r Docs/research/csgn_v0_4_1/requirements-reference.txt`); its complete timing/log was not saved locally, so it is not assigned an invented duration. The later editable-install log and final pip freeze capture the resulting environment.

| Command | Exit | Seconds | Log |
|---|---:|---:|---|
| `.\.venv-v041\Scripts\python.exe Docs/research/csgn_v0_4_1/verification/check_math_v0_4_1.py --output outputs/v0.4.1/m0-m2-20260919/reference_math.json` | 0 | 5.393 | `reference_math.log` |
| `.\.venv-v041\Scripts\python.exe Docs/research/csgn_v0_4_1/scripts/environment_probe.py --output outputs/v0.4.1/m0-m2-20260919/environment.json` | 0 | 1.988 | `environment.log` |
| `.\.venv-v041\Scripts\python.exe -I scripts/verify_handoff_subset.py` | 1 | 0.089 | `subset_initial.log` |
| `.\.venv-v041\Scripts\python.exe -I -B scripts/verify_handoff_subset.py` | 1 | 0.088 | `subset_after_bytecode_cleanup.log` |
| `.\.venv-v041\Scripts\python.exe -I -B scripts/verify_handoff_subset.py` | 0 | 0.089 | `subset_verified.log` |
| `.\.venv-v041\Scripts\python.exe -B -m pytest -q --junitxml=outputs/v0.4.1/m0-m2-20260919/tests_initial.xml` | 0 | 4.182 | `tests_initial.log` |
| `.\.venv-v041\Scripts\python.exe -B -m pytest -q --junitxml=outputs/v0.4.1/m0-m2-20260919/tests_reviewed.xml` | 0 | 4.047 | `tests_reviewed.log` |
| `.\.venv-v041\Scripts\python.exe -B -m pytest -q --junitxml=outputs/v0.4.1/m0-m2-20260919/tests_integrated.xml` | 0 | 5.386 | `tests_integrated.log` |
| `.\.venv-v041\Scripts\python.exe -B -m neuroplastic_v041.experiments.smoke --output outputs/v0.4.1/m0-m2-20260919/smoke_initial` | 0 | 8.531 | `smoke_initial.log` |
| `.\.venv-v041\Scripts\python.exe -m pip install -e .[test]` | 0 | 4.625 | `editable_install.log` |
| `.\.venv-v041\Scripts\python.exe -B -m pytest -q --junitxml=outputs/v0.4.1/m0-m2-20260919/tests_final.xml` | 0 | 5.742 | `tests_final.log` |
| `.\.venv-v041\Scripts\python.exe -B -m neuroplastic_v041.experiments.smoke --output outputs/v0.4.1/m0-m2-20260919/smoke_final` | 0 | 7.892 | `smoke_final.log` |
| `.\.venv-v041\Scripts\python.exe -I -B scripts/verify_handoff_subset.py` | 0 | 0.087 | `subset_final.log` |
| `.\.venv-v041\Scripts\python.exe -B Docs/research/csgn_v0_4_1/verification/check_math_v0_4_1.py --output outputs/v0.4.1/m0-m2-20260919/reference_final.json` | 0 | 3.542 | `reference_final.log` |
| `.\.venv-v041\Scripts\python.exe -B outputs/v0.4.1/m0-m2-20260919/verify_artifacts.py` | 1 | 0.293 | `artifact_verification.log` |
| `.\.venv-v041\Scripts\python.exe -B outputs/v0.4.1/m0-m2-20260919/verify_artifacts.py` | 0 | 0.284 | `artifact_verification_final.log` |

Read-only discovery also executed `git status --short --branch`, `git branch --show-current`, `git rev-parse HEAD`, `git worktree list`, `git remote -v`, scoped `rg --files`/`Get-Content`, `Get-FileHash`, CIM CPU/memory inventory and `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`. `git diff --check` passed. The implementation was staged by explicit paths and committed with `git commit -m "Implement CSGN v0.4.1 M0-M2 contracts and bounded lifecycle"` (exit 0). No remote mutation command ran. Scoped agent tests and their initial failure are described below; they are not additional independent experiments.

## 4. Tests

Final implementation suite: **163 passed, 0 failed, 0 skipped**, in 4.76 seconds pytest time. Final reference rerun: **24 groups / 19,438 local cases**, all passed; earlier reference rerun also passed. Final package check: 47 selected files, no errors. Test XML and logs are retained. M0/M1 have separate schema-valid manifests/results.

Coverage includes independent NumPy parity, saturated inward correction, FP32/FP64 numerics, nonfinite/invalid inputs, no-op and budget behavior, independent batch state, full recurrent graph aggregate identity, null/zero-availability/admission, exact singular-value norm enforcement and frozen-cycle contraction, conditional recurrent write/query derivatives, causal timing/reordering/expiry/generation and scale identity, target-leakage allowlists, rejection isolation and cold fast-state/retrieval guards. Central differences exclude support switches and max/min ties; stable active-face zero derivatives are tested separately. Ordinary autodiff is conditional on the selected smooth routing branch, not a gradient through hard support changes.

Preserved failures: the first graph fixture had `1 failed, 14 passed`: full and detached feature-write derivatives were identical because seven written edges did not overlap the next query. The finite-difference equality itself passed. A declared all-node fixture established overlap, with unchanged tolerances; production routing was unchanged. Exact failure text is in `graph_initial_failure.log`. Strict subset checks twice failed with `Unexpected: reference/__pycache__/math_reference.cpython-313.pyc` / `torch_reference.cpython-313.pyc`; generated files were preserved in evidence and removed individually, and all later Python runs use `-B`. No released bytes changed.

No GPU parity, trained-seed comparison, formal certificate or advanced subsystem check is claimed. Separate scoped final suites included 90 plasticity, 15 graph, 9 delta, 22 evidence and 15 feedback-math tests; these are subsets of the 163, not counts to add to it.

## 5. Results, controls and resources

Two task seeds (100,101), one fixed random graph initialization (0), no theta-training seeds. Each context had four support associations, one support pass and lawful random cue/context vectors. Each cold condition used eight query episodes, 64 correlated queries per context. No confidence interval is estimated from these correlated draws.

| Task seed | Model | Context | Cold SSE before sleep | Cold SSE after sleep | Cold accuracy after sleep |
|---:|---|---|---:|---:|---:|
| 100 | delta | old A | 0.01000000 | 0.00665754 | 1.000000 |
| 100 | delta | new B | 0.01000000 | 0.00492240 | 1.000000 |
| 101 | delta | old A | 0.01000000 | 0.00557866 | 1.000000 |
| 101 | delta | new B | 0.01000000 | 0.00532362 | 1.000000 |
| 100 | graph | old A | 0.01000042 | 0.01002067 | 0.000000 |
| 100 | graph | new B | 0.01000176 | 0.00992987 | 0.078125 |
| 101 | graph | old A | 0.01000017 | 0.00999665 | 0.171875 |
| 101 | graph | new B | 0.00999944 | 0.00997180 | 0.093750 |

All four shadow candidates passed the supplied empirical development screen. Each received four Adam steps, 32 replay examples and 256 validation-example evaluations (8 episodes for reference plus 8 for candidate). Graph seed100 old-context error worsened slightly; the configured old-loss allowance of .01 admits that change. Acceptance is not a lifetime or efficacy certificate. Delta dominates these tiny fixed-feature cold scores; its engineered tensor-product keys and resources differ, so this is not a trained architecture comparison.

Final smoke: 6.107 seconds internally, 7.892 seconds command wall. Both smoke command runtimes total 16.423 seconds against the 900-second overall smoke budget; each stayed below 180 seconds. No parameter sweep or hyperparameter search was run. Peak process working set: 319,533,056 bytes, including interpreter/libraries and both models. Tensor activation and Python-object peaks are not fully decomposed; no zero placeholder substitutes for that missing decomposition.

Per graph lifetime: 14,336 state tensor bytes; 24,844 theta bytes; 4,096 buffer bytes; 19,968 topology bytes; 84,840 peak charged snapshot/pending bytes; replay peak 1,280 bytes. Delta: 12,312 charged state bytes, no trainable shared theta, 563 snapshot bytes, replay 1,280 bytes. Graph sleep optimizer/gradient tensors: 4,100/2,048 bytes; delta: 8,196/4,096 bytes, with independent shadow copies additionally reported. Replay data reads were 4,096 tensor bytes/model/lifetime. These components overlap transient process residency and must not be mistaken for an exact additive process peak.

Graph sleep fit took approximately .14 seconds/lifetime and empirical validation .70 seconds; delta fit .009 seconds and validation .05 seconds. Full timings are in lifetime JSONs. Dense graph execution computes 512 logits per decision and 128 source/head messages per inner round; only 128 candidates and up to 64 edges are selected/executed per decision. Support routing events and their queue service are retained separately. No sparse-speed or matched-latency claim is made. Checkpoint artifacts were not created; config, RNG derivation, task hashes and exact source are retained for reconstruction.

## 6. Diagnostics

All 16 graph support writes satisfied the executed local descent/budget checks. There were zero observed endpoint failures, lost pending credits, overflow or replay evictions in the smoke. Passive decay is separately logged (zero movement under this retention profile); intentional write counts and no-progress/projection statistics remain per event.

**6/16 writes had negative future-query utility**, ranging from -0.000130058732 to 9.7883807e-05 SSE improvement. Thus local descent did not guarantee later improvement. Both branches began from identical post-decay state and used the same scorer-only queries; probe states/labels were not returned to the learner.

All 16 sampled designs had free-coordinate rank 8. Storage-only half-SSE floors ranged from 0.0014452198 to 0.00377430913, with maximum primal–dual gap 2.35e-11. This is numerical evidence under each fixed captured design and box, not a downstream irreducible floor. Singular values, condition numbers, active bounds, projected-gradient residuals and solver status are retained. Per-step allowances were excluded from the floor calculation.

Fourteen of 16 immediate write/read probes had zero writable-support overlap/design cosine under the bounded demand scheduler. The other overlaps were roughly .35–.38. This is a concrete address/service issue to diagnose before shared-feature training, not proof that every future read fails. Exact frozen-cycle q was approximately .875, within its declared numerical condition; no global adaptive contraction is claimed. Cold tests kept both residual tiers off on every query and verified unchanged S. Transplant, erasure and rule-transfer diagnostics are `not_run`.

## 7. Deviations and open specification questions

- **D01, instruction precedence:** prior root preparation-only restriction is superseded by this fresh explicit implementation request. No new permission flow was inferred.
- **D02, I6/§9 byte caps:** shipped config capped records but omitted byte caps. Implemented explicit charged-byte limits of 16,000,000 pending and 32,768 replay; counts plus measured process memory also constrain Python overhead. These are engineering choices, not mathematical revisions.
- **D03, profile routing count:** narrative's 64-sender example is an upper example; CPU config uses 16. Actual candidate/computation counts are distinguished. Region summaries are recomputed; WM is an explicit zero-capacity no-write profile.
- **D04, diagnostic sampling:** initial smoke examined all eight writes despite its .1 rate. Initial outputs are retained and interpreted with that discrepancy. Final config explicitly sets rate1; runner honors rate using an independent diagnostic RNG, with the same eight-call cap. Final predictions match initial ones because diagnostics are isolated. No released config was edited.
- **D05, clocks/validation:** review caught warm-probe queue timestamps moving backward and eight validation episodes being concatenated into one recurrent sequence. Both were fixed before the first smoke; each final episode resets once. Generation/timing/expiry controls remain tested.
- **D06, I7 numerical interpretation:** full rank coexists with substantial bounded snapshot floors and poor query behavior. Options are improving feature scale/geometry or routing service, not assuming rank guarantees capacity. Source/reference/paper equations were not changed.
- **D07, comparison scope:** delta has engineered 128-dimensional context keys and a stronger unconstrained-by-graph-step delta update; same feedback/sleep exposure, different bytes and latency. Static/replay-only trained baselines are deferred to M3.
- **D08, incomplete full T1:** post-cold supersession and repeated horizons are deferred; this run covers the requested minimum old/new–write–sleep–clear–recall lifecycle.

- **D09, config artifact serialization:** the initial manifest correctly hashes the original input config, but saving parsed JSON changed LF to CRLF. The first artifact byte-equality assertion failed and remains logged. Supplemental `smoke_initial/config_input_original.json` matches the original input hash; `config_serialization_note.json` records both byte hashes and parsed equality. The final config is byte-identical to its saved copy. The corrected artifact verifier checks original input bytes plus parsed-copy equality and passed 139 checks. No numerical result or release byte was changed.

No confirmed scientific contradiction between the authoritative paper and central reference equations was found. Unresolved research questions: how much demand scheduling causes address drift; whether feasible feature scaling can lower the storage floor; whether shared learned representations improve later-query utility; and whether any gain survives resource-matched delta/replay baselines. Preserve the negative findings while resolving these questions.

## 8. Claim assessment

H1 **untested**: no shared-representation training. H2 **untested as an efficacy claim**: slow fitting and cold evaluation execute, but this small fixed-feature result lacks trained and matched replay controls. H3 **untested**: topology is fixed. The implementation and local contract evidence support only their tested behaviors. A smoke test is not architecture validation.

## 9. Next bounded action

Before any M3 run, review the observed support discontinuity against the declared queue policy. The already implemented bounded derivative/control reproducer is:

```powershell
.\.venv-v041\Scripts\python.exe -B -m pytest tests/v041/test_graph.py -q -k recurrent_write_query_outer_gradient
```

It uses CPU and should take under 10 seconds (cap 30 seconds); stop on nonfinite state, support-boundary mismatch or gradient disagreement. A pass establishes the conditional gradient plumbing, which already passes in the full suite; it does not resolve the production scheduler's drift. The next new experiment should add a same-observation, same-budget queue/service control and inspect downstream utility before training or enlarging the model. That new experiment is not implemented or launched in this slice; its exact training command must be specified only after implementation/review. No two-hour pilot is authorized by this report.

## 10. Artifact inventory

Primary local evidence: `outputs/v0.4.1/m0-m2-20260919/`. Final run manifest/result: `smoke_final/run_manifest.json` and `smoke_final/run_result.json`; aggregate milestone manifests/results and M0/M1 subdirectories are also schema-validated. `artifact_inventory.json` records paths, byte sizes and hashes for local artifacts, excluding itself and the verifier's own in-flight log.

| Final artifact | SHA-256 |
|---|---|
| run_manifest.json | a337a064aade9f6334fbc1ee0635fd6eb25afc0b68337c13170290a6d794d890 |
| run_result.json | 9a16b4d2f2ac26042fb0b01929017597dfa5f027312fb2f5a9c2767a524aa82f |
| source_manifest.json | 00adb34acdb20d29be8b48ef0af5199d56c7ce241bded7932444914bed165993 |
| source_snapshot.zip | bedee780931f8373208530f009a6bc5c01d8618f3eaa75924cd29bcd45d8d6da |

Both smoke runs retain exact source archives and tracked/untracked dirty patches; final dirty patch is empty. Per-lifetime JSON includes every phase, seed, sleep decision, write and sampled diagnostic, with partial artifacts retained. Reference reruns, original bytecode-check failures, graph fixture failure, final test XML, environment and pip freeze are preserved. No old checkpoint, secrets, credentials, private user dataset or historical payload was included; generated synthetic metrics/source stayed local and no bundle was uploaded.

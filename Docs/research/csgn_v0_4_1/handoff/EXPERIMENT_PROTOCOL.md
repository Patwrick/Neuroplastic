# Experiment protocol — v0.4.1

**No runs described below have been executed as integrated CSGN experiments.** The shipped check suite is local mathematics only. These protocols and configs are the implementation target, with any changed engineering choices recorded before interpretation.

## 1. Information boundary common to every task

An observation exposed to the agent contains a bounded cue vector, a legitimately observable context vector, an event type, and a decision identifier used only to match feedback (not a feature). The target, hidden association table, generator seed, transformation matrix, future reward, and acceptance labels are accessible only to environment/scorer logic. Do not pass an `info` dictionary with these fields through the model encoder. Unit tests inspect the observation schema and feedback timing.

A predict event precedes its label/target revelation. Once feedback becomes available, the write API receives it together with the corresponding captured snapshot. Query-only phases do not reveal answers or permit writes. Support and scored query targets are distinct arrays owned by different interfaces. Use independent RNG streams for task generation, environment events, model initialization, routing noise, replay, and evaluation; record seeds but never feed their values to the model.

Context labels may disambiguate tasks; they must not encode the task's answer table. Shuffle or redraw context descriptors independently of targets between lifetimes. Entire query episodes and training seeds, rather than correlated time steps alone, are units for uncertainty estimates.

## 2. T0 — deterministic mechanism fixtures

Purpose: prove plumbing before neural learning. Use the shipped analytic small matrices and known feasible targets. Test endpoint inward correction, exact A construction, and slow transfer with residual reset. T0 success is never the H1 result. An ideal orthogonal-key memory is not a trained graph.

## 3. T1 — learned cue/context association and consolidation

Default: cue dimension 16, context dimension 8, 8 target classes. A class is encoded by a fixed 8-dimensional vector: `target_scale * one_hot(class)` with target_scale=0.1 in the first pilot. The codebook is fixed and public; the context-to-cue class mapping is random, private to the task, and independent of cue/descriptor values. Class accuracy is nearest-code distance; training also logs raw squared error.

Generate normalized random cue vectors and independent normalized context descriptors from seeded task streams. Do not supply ideal orthogonal features to the H1 learner. A lifetime starts from a shared learned initialization and receives multiple context tables. Each context has 16 cue identities in the reference pilot. Support first predicts each assigned target and then reveals the corresponding value. Query phases revisit cues without feedback. Add distractor contexts, train the next context, re-query still-valid prior contexts, run sleep, clear all fast/WM/workspace state, and perform cold queries with answer retrieval and writes disabled.

Association capacity is swept; the initial setting is not presumed feasible. The hidden tables remain available to the scorer across sleep, but are never in model state. Split generators for meta-train, development, and confirmation. Within an episode support/query cues may match because T1 tests memory; across test episodes redraw cues/context assignments. Claims about unseen-input generalization require T3, not T1.

Reference sequence per lifetime: learn context A, warm query A, hold/distractors, learn B, query A and B, maintenance, cold query A and B, then a labeled supersession in one *distinguishable* context. Query old still-valid items separately from intentionally changed ones. More cycles are a horizon sweep; do not use resetting between every query to make a stateful task easier without labeling the condition.

## 4. T2 — delayed target and resource pressure

Use T1 targets revealed after delays from a declared set, initially 0, 1 and 4 events, then 16 and held-out delay ranges. Prediction time and feedback time are recorded. Preserve captured factors or sufficient recoverable data; changing features after the decision does not authorize using a new matrix as though it were the same snapshot.

Cap pending records (128 pilot), replay bytes, and resident hot identities. Count expiration, spill, version rejection, and unprocessed feedback. Never merge per-event snapshots into one edge trace unless an algebraic equivalence is shown. Sweep arrival density separately from delay. Evaluate loss against delay and actual pending bytes. Reward-only RL is not claimed here: labels are delayed, not absent.

## 5. T3 — structured rule transfer beyond memorization

Input x is 8-dimensional and embedded in the first eight positions of the 16-dimensional cue channel. Query inputs have norm at most one and are absent from support. A hidden context-dependent signed permutation T maps x to `target_scale * T x`, an 8-dimensional continuous target. Context descriptors are independent of T. This target is compatible with an 8-dimensional public output interface.

Begin with diagonal sign maps for pipeline diagnosis, then signed permutations. Use at least eight linearly independent support inputs; the pilot uses 16 support vectors and checks rank=8. Reject/redraw degenerate support *in the generator*, recording the count, rather than demand impossible arbitrary-map recovery. A well-specified diagonal family can require fewer observations, but that is a distinct task condition.

Test new x under the observed context, delayed revisits after distractors, and cold recall after sleep. Hold out transformation matrices and input samples; additionally hold out a map family for a clearly labeled OOD study. A fixed function of cue alone cannot solve different T at the same x. Report regression error, and do not reuse T1 nearest-code accuracy for continuous rule outputs.

Compare graph memory with a simple context-keyed delta memory and a clearly labeled least-squares task oracle. The oracle is a diagnostic and must not be used to generate hidden internal teacher targets for the main model without declaring that extra supervision.

## 6. T4 — memory transplantation and selective erasure

Choose a single trained shared theta checkpoint and a fixed compatible topology/descriptor/decoder version. Create donor and recipient with identical initialization but different private random T1 tables. Freeze theta while each learns/consolidates. Run four cold states: donor original, recipient original, recipient with donor S, and recipient with original S restored. Swap only slow values (and their representation metadata if a later quantized condition needs it), never theta/readout/task lookup. Both residual tiers, writes and episodic answers remain disabled; h/WM are reset identically. Predictions should track the source of S for demonstrated donor knowledge.

If donor/recipient have not learned or disagree only weakly, the transplant is non-diagnostic; report that, not a causal pass. Measure donor-target loss and recipient-target loss separately, the degree of donor tracking, and effects on unrelated contexts. These metrics are empirical, not a demand for perfect transplant accuracy from an imperfect model.

For erasure, select edges/blocks using development data, then remove them in a cloned model. Compare equal-count random removals and matched magnitude/usage removals; charge the selection data. Report affected versus unaffected query losses and total damage. A generic output collapse is not selective localization.

## 7. T5 — leakage and no-information controls

Control A draws a new independent target class at every scored query with no predictive information in the available history. Eight equiprobable classes imply chance expected accuracy 1/8; report a valid interval rather than treating a small random deviation as failure. Do not reuse stable cue–label maps and expect nonlearning: those are legitimately memorizable.

Control B randomizes support feedback independently of a stable hidden truth used for query scoring. Control C removes writes entirely. Control D swaps decision-to-feedback pairing deliberately in a labeled diagnostic. Success only with leaked task seeds, hidden target fields, or incorrect target timing is invalid evidence. Automated field/time assertions are required in addition to behavioral controls.

## 8. T6 — write utility, reachability and address stability

On preselected diagnostic episodes, clone the complete state/RNG before one proposed write and compare written/unwritten branches on the same future query list without additional learning. Log U=loss(no-write)-loss(write), captured local loss change, and protected-old loss change. Do not use this query list to tune the same run's hyperparameters. If evaluating closed-loop actions, trajectories can diverge; common random numbers do not make the environment histories identical, so state the intervention's interpretation.

For the same snapshots, solve the storage-only box problem. Report objective upper/lower bounds, gap, rank and solver status; do not call a high unconverged residual an irreducible floor. The per-step box is a separate optimization-rate diagnostic. Log the fraction of unreachable/ill-conditioned/stale targets and correlate with write utility without claiming correlation proves cause.

Compare `state_dependent` versus `stable_address` with the same information and dimension. A separate `stable_write_features` variant changes message coding and needs its own name. Sweep the sufficient workspace contraction setting within valid norm bounds, not by silently disabling stability checks. Report actual norms and gains, not only configuration labels.

## 9. Baseline matrix and fairness

Primary models:

- `delta_memory`: task-grounded encoder plus bounded normalized delta memory, independent retention, same available labels and context.
- `static_graph`: same encoder/graph training budget, no online plastic writes.
- `replay_only`: no fast writes; slow fitting at maintenance with the same support/replay information and replay compute.
- `graph_fast`: v0.4.1 writes, no slow consolidation.
- `graph_sleep`: v0.4.1 writes plus slow-only sleep/cold recall.

Ablations: old absolute-sum reserve, detached feature-to-write outer path, no write, stable addressing, contraction strength, retrieval enabled, and—only later—rewiring. Do not conflate a test-time disabled write with a separately trained static baseline. A no-sleep model's cold performance is diagnostic of where its learning resides, not a fair claim that it was trained to consolidate.

Use two explicit comparison tracks. **Shared-feature mechanism track:** same frozen encoder checkpoint, equal task/feedback exposure, memory-byte accounting. **End-to-end track:** separately trained models with equal training and hyperparameter-search budgets, tuned only on development data. Report parameters and optimizer/activation state as well as edge counts. Match memory and wall-clock frontiers separately when simultaneous equality cannot be achieved. Give the delta baseline an equally specified slow/replay consolidation option when testing H2, not only a fast-only comparator.

## 10. Evaluation and reporting

Pilot defaults are 3 independent training seeds and 64 held-out episodes per task. This diagnoses execution and variance; it is not automatically sufficient power. Confirmation proposes 10 or more independent training seeds, at least 512 held-out episodes per task per seed, and preregistered resource/effect thresholds. Revise sample sizes from pilot variance and cost *before* opening confirmation data.

Report per-seed raw metrics, paired differences on shared task draws, uncertainty clustered at trained-seed/task-episode levels, and all stopped/failed runs. Pooling thousands of time steps from one model does not create thousands of independent trained models. Plot learning/retention curves rather than just endpoint maxima.

Primary hypotheses: H1 learned-query improvement from graph writes, H2 useful cold recall beyond the appropriate replay comparator, H3 optional net value of rewiring. Set minimum effect size, tolerated old-task degradation and resource ratios in a preregistration record. The `confirmation_plan.json` leaves these values null until that decision is recorded. Null means **not launchable**, not zero tolerance.

No hidden “success” threshold may be selected after seeing confirmation results. Return negative evidence and proposed revisions separately from the preregistered analysis.

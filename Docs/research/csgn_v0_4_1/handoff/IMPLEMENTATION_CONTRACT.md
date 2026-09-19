# Implementation contract

Normative mathematics is in the paper. This document specifies an initial engineering realization. IDs below map to tests in CONTRACT_TEST_MAP.md. Shapes are unambiguous: B is independent lifetimes, N columns, E reserved edges, d workspace width, dm message width, H heads, and V output-code width.

## I0. Package and provenance boundary

Import the released package without altering its shipped files. New code belongs in an isolated namespace and new runs in versioned run directories. Record package manifest digest, paper SHA-256, implementation commit and dirty diff hash. A dirty-tree experiment is allowed only with the exact diff retained and explicitly identified. Do not use an old result as evidence for new code.

## I1. State ownership and public interfaces

Suggested modules (names may adapt to repo conventions):

- `state.py`: AgentState, ModelVersion, SlotIdentity and immutable snapshots.
- `routing.py`: exact regional/local scoring, support and admission.
- `graph.py`: typed messages, receiver envelopes, inner workspace and readout.
- `plasticity.py`: endpoint set, intervals, projected writes, decay and diagnostics.
- `feedback.py`: causal pending records, pin/spill/expiry/version policy.
- `memory.py`: bounded working memory and finite episodic replay store.
- `consolidation.py`: slow-only fitting, candidate snapshot, validation and atomic publication.
- `tasks/`: the protocol-defined tasks, not an environment with target labels embedded in observations.
- `baselines/`: simple delta memory, static graph and replay-only learner.
- `experiments/`: seeded training, paired evaluation, interventions and logging.

Suggested function contracts, not existing executable API:

```text
predict(state, observable, permitted_context, mode) -> prediction, next_state, write_snapshot
receive_feedback(state, record, target, evidence) -> next_state, WriteMetrics
consolidate(state, replay, validation_policy) -> candidate_or_same_state, CommitReport
cold_evaluate(snapshot, query_stream, protocol) -> EpisodeMetrics
snapshot(state, include_rng=True) -> IndependentSnapshot
restore(snapshot) -> AgentState
```

`AgentState.S`, `Pa`, and `Pu` have independent ownership per lifetime. A dense correctness implementation may use `[B,E]` tensors, but must count them and label the layout as dense storage. Topology is `[E]` source, target, head, generation with globally fixed slot ordering; batching may share topology only when explicitly identical. `h` is `[B,N,d]`; working memory and pending records are per lifetime. No averaging of synaptic states across B. Offline shared `theta` may be trained across B, while per-lifetime state follows its own episode graph.

Snapshots deep-copy non-autograd state and RNGs for interventions. Training-time feature tensors must preserve their outer autograd lineage; copying and detaching are different operations. Synchronous deployment snapshots may be detached because theta is frozen. Avoid aliasing old state with in-place writes, especially across a shadow candidate, ablation branch, or batch element.

## I2. Endpoint feasibility (paper 4, 18, 19)

For each edge test `S`, `S+Pa`, `S+Pu`, and `S+Pa+Pu` against `[-wmax,wmax]`, and `abs(Pa)+abs(Pu)<=pmax`. This is **not** the old `abs(S)+abs(Pa)+abs(Pu)` reserve.

For the chosen residual p and fixed other residual o:

```text
R = pmax - abs(o)
L = max(-R, -wmax-S, -wmax-S-o)
U = min( R,  wmax-S,  wmax-S-o)
offset = S+o
zminus = offset+p
lo = max(offset+L, zminus-allowance)
hi = min(offset+U, zminus+allowance)
```

First validate the unchanged endpoints and starting chosen residual. `allowance>=0`, and the sum over the write set must not exceed the recorded global intentional-write budget. Passive decay is logged separately. Check finite arithmetic before clipping; NaN must not pass an inequality by accident. Floating-point tolerance repairs must be tiny, explicit and counted, not a way to admit grossly infeasible states. Empty intervals are errors; zero-width intervals are valid.

## I3. Exact small routing and actual graph computation

Reference sizes come from the config. Give each column a fixed descriptor known to both baseline and intervention copies. State-dependent local keys use the paper's query–key interaction, not a shared additive context offset. Evaluate all region scores and local candidate scores at small scale; charge the work.

For real outgoing candidates, exclude availability zero *before* top-k. Use logits plus log availability and keep the null route in the normalization. Apply receiver-capacity admission without redistributing rejected mass. Keep scored, executed and writable sets distinct. Freeze chosen support, gates, weights and context for all K inner rounds. During meta-training, preserve continuous derivatives through selected logits/gates; do not claim exact differentiation of discrete support switches. Deterministic ties use a documented stable rule. Changes in topology/admission are not transparent cache operations.

Use typed bounded messages and the weight-independent envelope `Zj=max(1,wmax*sum(ae))`. In float64 unit tests, form an explicit A matrix and compare `A@z + nonwritable_contribution` against the actual graph aggregate. Runtime should use gather/scatter or segmented sparse products rather than materialize E-by-N-by-d arrays. A dense tiny reference is allowed for parity checks but must not support a sparse-speed claim.

Update only the declared sender/receiver/interface set. Unupdated columns are boundary values, not silently cleared. For protected routes, build and test static connectivity, then separately log executed time-respecting paths and queue service. Four internal rounds and one environmental step are distinct time units.

Use operator-norm constraints satisfying Equation (12) in the reference. At small dimensions, exact singular-value calculation is acceptable; at larger sizes a rigorous upper bound or explicitly empirical alternative must be declared. Logging a few power iterations does not certify an upper bound. The contraction theorem applies with identical frozen boundary/input conditions, not to the entire adaptive loop.

## I4. Target and snapshot contract

The reference prediction is a designated receiver aggregate or a fixed linear decoding of designated aggregates. A trainable nonlinear readout can report additional task predictions, but its loss is not the locally guaranteed quadratic. Write targets come from externally revealed value codes or a fixed target encoding of later observations. All observation dicts must pass a target-leakage allowlist test.

For a decision snapshot retain selected edge identities/generations, source message tensors, gates, receiver budgets, decoder convention, input/context/feature versions, prediction time and feedback-available time. Construct columns of A from those exact captured factors. When writing only a subset/tier, subtract nonwritable contributions from the target using the values explicitly fixed for this update, including post-decay other-tier state. Test identity against the captured graph output before applying the update.

Apply `eta=beta/(sum(A*A)+eps)` and the exact interval projection. Recompute captured loss before/after and the descent upper bound in diagnostics. A zero matrix or beta=0 means a logged no-op. The inner theorem does not license refreshing A halfway through a step. Delayed records can refer to currently inactive edges. On generation/version mismatch, reconstruct from permitted retained observations or reject and count lost credit. Never apply a stale record to a newly reused slot.

## I5. Differentiation contract

For short meta-training episodes, keep continuous paths through A(theta), selected gates, feature coding, initial slow parameters when meta-learned, rate normalization, and functional projection. Compute the partial z-gradient analytically as `A.T@(A@z-b)`; this does not require detaching A from theta. Do not put the training-time write inside `no_grad`, call `.item()` on a learned rate used in the update, or round-trip tensors through NumPy.

The exact reference is conditional on fixed discrete support and a locally constant max/min active set. Finite differences use central perturbations away from ties, clipping transitions, and top-k changes. Test active-face zero derivatives separately. Where support changes, report nondifferentiability or a named surrogate; never adjust the finite-difference tolerance until a wrong gradient passes.

`detached_feature_write_path` removes continuous feature-to-write paths as an ablation while leaving ordinary direct query paths intact. It is not the same as freezing all theta learning. Targets are externally given/frozen in the reference; learned internal teachers are deferred.

Offline theta changes occur between training episodes or declared unroll boundaries, not in the middle of a deployment lifetime without version handling. Store episode/task/train seeds separately. Recurrent state is reset only at declared boundaries; no arbitrary resets solely to improve a score.

## I6. Memory and consolidation

Both residual tiers and WM are bounded and per lifetime. Episodic records have finite byte/capacity budgets, validity metadata, time and feature version. The first clean synthetic tasks mark their feedback directly admissible. Keep Pu present and test its boundary, but do not claim an adversarial isolation guarantee from two buffers.

At reset-boundary maintenance, pause writes, fix event cutoff, create an independent shadow state, and fit S with Pa=Pu=0. Initially freeze theta and topology. Slow optimization is a nonconvex full-graph training procedure; it has no Proposition 5 convergence guarantee. Project S to its valid bound in the zero-residual candidate and log actual losses. Match its replay exposure and optimizer budget in the replay-only comparator.

Exact transfer is optional initialization; it is not a substitute for slow-only fitting/evaluation. Endpoint feasibility under exact transfer is tested. Quantization is disabled initially; any later quantized compensation must remain forward-active and pass the same endpoint checks.

Cold evaluation fixes Pa=Pu=0 for the *entire* rollout, disables writes, resets/standardizes h and WM, disallows episodic answer retrieval, and provides only lawful context. Predictions must not call an implicit replay/rewrite helper. Snapshot state before/after evaluation to verify immutability except permitted transient workspace advancement. Slow storage must remain identical.

Default publish semantics: synchronous episode-boundary candidate accepted or rejected as a whole. A failed candidate cannot partially modify live topology, RNGs, residuals or controller state. Tests must prove rejection isolation, pending-record reconciliation, and generation mismatch handling. Asynchronous maintenance is deferred.

## I7. Required diagnostics and interventions

- `write_utility`: deep-cloned with/without-write branches, identical query stream and RNG conditions; no evaluation labels leak into continued live training.
- `reachability`: solve the storage-only box, not the per-step box. Report solver status, rank, singular values, actual upper bound, numerical dual lower bound and gap. If not converged/tight, do not call the achieved residual an irreducible floor.
- `address_drift`: compare write/read key/message similarity and support overlap for the same lawful query.
- `transplant`: same theta/G/descriptors/decoder, only S swapped; all fast/WM/EM answer channels disabled. Different theta checkpoints are not directly transplant-compatible.
- `erasure`: select using development data, compare equal-budget random and magnitude/usage-matched groups; distinguish global model damage from selective memory loss.
- `negative_controls`: fresh independent labels versus stable random mappings are different tasks. Only the former lacks learnable label information.

## I8. Logging and resource accounting

Every run has config hash, code SHA/dirty diff, seed, task generator/split hash, environment, state/operation bytes, replay/validation counts, elapsed wall time, stop reason and artifacts. Never write only aggregate best scores. Preserve per-seed/per-task results and uncertainty. Confirmation data are held out from tuning and diagnostic edge selection.

The report distinguishes an algebra check, implementation smoke, learned pilot, independent comparison and formal conditional statistical claim. `not_run`, `failed`, `aborted`, and `inconclusive` are supported values. Do not replace missing measurements by zero.

## I9. Inference interface remains environment-agnostic

Keep the observation/action/feedback interface separate from the learning core. Do not build Minecraft, robot control, vision training pipelines, or pretrained-model integration in the first slice. Those would obscure the mechanism result and are not prerequisites for it. The first tasks are synthetic, locally generated, causal and versioned.

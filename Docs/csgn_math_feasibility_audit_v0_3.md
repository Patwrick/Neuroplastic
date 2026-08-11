# CSGN v0.2 Mathematical and Feasibility Audit
## Design basis for Cortical Synaptic Graph Networks v0.3

**Date:** 2026-08-11  
**Scope:** Equation-by-equation audit, implementation comparison, scale analysis, failure analysis, and corrected mathematical requirements.  
**Audited artifact:** `Docs/csgn_paper_v0_2.md`  
**Implementation inspected:** `csgn/model/plastic_graph_core.py`, `csgn/model/sleep_controller.py`, and the committed toy experiment scripts/results.

---

## 1. Executive verdict

CSGN is **not mathematically impossible**, and its main ingredients have credible precedents: meta-trained plasticity, test-time memory, replay, dynamic sparse training, and recurrent graph computation. The core research direction remains worth testing.

However, **v0.2 is not yet a closed mathematical specification**, and the target-scale system is **not implementable as written**. Several defects are blocker-level rather than cosmetic:

1. **Consolidation is not function preserving.** Equations (20)–(21) generally cause an abrupt change in effective synaptic strength after every sleep cycle.
2. **The target-scale state budget is understated by more than an order of magnitude.** At 100B edge slots, the explicitly proposed edge state is about **1.25 TB** before node states, activations, replay memory, optimizer state, redundancy, or safety snapshots.
3. **Rewiring is globally expensive.** Equations (26)–(27) imply an all-pairs coactivation table or a softmax over all nodes for every new edge.
4. **Sparse activation does not imply sparse execution.** The paper omits node selection, key/query projection, recurrent update, trace update, and random scatter costs; the present code computes over all nodes and all edge slots.
5. **The plasticity rule is not bounded.** Eligibility and fast traces can grow as inverse products of their decay gaps, and a single scalar modulation signal assigns the same delayed learning sign to every eligible edge.
6. **Utility is not causal importance.** The absolute update magnitude in Equation (17) protects surprising or poisoned synapses whether they helped or harmed performance.
7. **The topology can lose semantics and connectivity.** No invariant prevents duplicate edges, self-loops, disconnected regions, receiver hubs, or repeated pruning of newborn edges.
8. **The current graph-core experiment does not test the stated hypothesis.** Its shared neural modules are random and run under `no_grad`; the graph core is not meta-trained. The committed binary-task runs remain approximately at chance.

These findings do **not** prove that CSGN must fail. They mean that v0.2 cannot support the paper's current feasibility and scaling claims without substantial revision.

---

## 2. Severity definitions

- **BLOCKER:** Contradicts a central claim, makes the target implementation infeasible, or causes an unavoidable semantic discontinuity.
- **CRITICAL:** Creates likely instability, forgetting, routing collapse, or invalid credit assignment unless corrected.
- **MAJOR:** Underspecified, incorrectly costed, or missing an important invariant or control.
- **MINOR:** Notational, dimensional, or presentation defect that is straightforward to repair.
- **VALID WITH CONDITIONS:** Mathematically sound after stated assumptions are made explicit.

---

## 3. Equation-by-equation audit

## 3.1 Cortical graph core: Equations (1)–(11)

### Equation (1): activity score

\[
a_i^t=\lVert h_i^t\rVert_2
\]

**Verdict: MAJOR.** The expression is valid, but norm-only top-k routing creates several failure modes:

- At initialization, equal or near-equal states create deterministic tie selection.
- High-norm nodes receive a persistent routing advantage, producing rich-get-richer collapse.
- A node can be semantically important but low norm.
- Global top-k requires an \(O(N)\) scan and selection every step.
- The rule has no regional quotas or load-balancing term.

**Required correction:** use hierarchical routing, learned router logits, stochastic tie-breaking during training, regional capacity constraints, and a load-balancing objective. A global scan must not be required at target scale.

### Equation (2): shared message transform

\[
m_i^t=\Phi(h_i^t,c^t)
\]

**Verdict: CRITICAL expressivity limitation.** Every outgoing edge from source \(i\) carries a scalar multiple of the **same message vector**. Therefore, all outgoing messages from a source are collinear at a given time. A source cannot send different content to different destinations; it can only change amplitude.

**Required correction:** use a small number of shared message heads,

\[
m_{i,r}^t=\Phi_r(h_i^t,c^t),\qquad r\in\{1,\ldots,H\},
\]

and give each edge a compact type/head identifier. This raises the outgoing message rank from one to at most \(H\) without requiring a full edge-specific matrix.

### Equations (3)–(5): dynamic gating

\[
q_i^t=W_Qh_i^t,\quad k_j^t=W_Kh_j^t,
\]

\[
\ell_{i,s}^t=\frac{(q_i^t)^\top k_j^t}{\sqrt{d_k}}+b_{i,s},
\]

\[
g_{i,s}^t=\operatorname{softmax}_s(\ell_{i,s}^t).
\]

**Verdict: MAJOR.** The dimensions are valid once \(W_Q,W_K\in\mathbb{R}^{d_k\times d}\) are stated. The missing issues are:

- \(b_{i,s}\) is another per-edge slow parameter and is absent from the memory count.
- Softmax assigns nonzero mass to every slot; routing is not actually sparse within the outgoing slots.
- The outgoing mass must sum to one even when the correct action is to send nothing.
- Target-key gathers are irregular and expensive.
- There is no receiver capacity or in-degree constraint.

**Required correction:** add a null route, use entmax or hard top-r routing, cache low-dimensional target keys by region, and impose receiver-capacity penalties or normalization.

### Equations (6)–(7): effective strength

\[
\widetilde w_{i,s}^t=w_{i,s}+\alpha_{i,s}^tf_{i,s}^t,
\qquad
c_{i,s}^t=g_{i,s}^t\widetilde w_{i,s}^t.
\]

**Verdict: CRITICAL unless bounded.** The units are consistent, but neither \(\alpha\) nor \(f\) is intrinsically bounded in v0.2. Equation (18) makes \(\alpha\) unbounded under softplus. The product \(\alpha f\) can dominate slow memory and routing.

**Required correction:** use

\[
\alpha_{i,s}=\alpha_{\max}\sigma(\beta_{i,s})
\]

and a bounded fast-state update. At scale, \(\alpha\) should usually be shared by block, edge type, or region rather than stored per edge.

### Equation (8): incoming aggregation

\[
r_j^t=\sum_{i\in A_t}\sum_{s:\tau(i,s)=j}c_{i,s}^tm_i^t.
\]

**Verdict: BLOCKER for stability at scale.** Fixed out-degree does not imply bounded in-degree. Rewiring can create receiver hubs, making the incoming norm scale with the number and magnitude of incoming edges.

A stable receiver-normalized form is

\[
Z_j^t=\varepsilon+\sum_{(i,s)\to j}\chi_i^t\lvert c_{i,s}^t\rvert,
\]

\[
r_j^t=\frac{1}{Z_j^t}\sum_{(i,s)\to j}\chi_i^tc_{i,s}^tm_{i,\kappa(i,s)}^t.
\]

If \(\lVert m_{i,r}^t\rVert\le M\), then \(\lVert r_j^t\rVert\le M\). This supplies a direct bound independent of in-degree. The total incoming magnitude can be passed separately as a bounded feature if amplitude information is needed.

### Equation (9): recurrent state update

\[
h_j^{t+1}=\Psi(h_j^t,r_j^t,x_t,\mathrm{WM}^t).
\]

**Verdict: BLOCKER as a mathematical specification.** \(\Psi\) is unconstrained, so no bounded-state, contraction, Jacobian, or Lyapunov property follows. A generic recurrent MLP can explode, collapse to a fixed point, oscillate, or become chaotic after plastic updates.

**Required correction:** specify an explicitly bounded recurrent cell and enforce a stability envelope. At minimum:

- bounded candidate states,
- gated convex state updates,
- spectral normalization or another Jacobian constraint,
- online estimates of the dominant Jacobian norm,
- rollback when a sleep update violates the envelope.

Strict global contraction is not required and may erase useful persistent modes, but the model needs a declared near-critical operating range and a measurable invariant.

### Equations (10)–(11): pooling and readout

**Verdict: MAJOR.** Pooling only active nodes is valid, but mean pooling loses node identity and may erase sparse localized information. The current implementation averages **all** nodes, not the active set. A query-conditioned, region-aware readout is preferable.

---

## 3.2 Plasticity: Equations (12)–(23)

### Equation (12): pre/post features

**Verdict: VALID WITH CONDITIONS.** The feature dimension must be shared and stated. Features should be normalized or bounded before their dot product enters an accumulating trace.

### Equation (13): eligibility trace

\[
e_{i,s}^{t+1}=\lambda_e e_{i,s}^t+(p_i^t)^\top s_j^t.
\]

**Verdict: BLOCKER for bounded dynamics.** If \(\lvert z_t\rvert\le Z\), where \(z_t=(p_i^t)^\top s_j^t\), then

\[
\lvert e_t\rvert\le \frac{Z}{1-\lambda_e}.
\]

For \(\lambda_e=0.95\), this is a factor of 20 amplification before the fast trace is updated. If unnormalized 64-dimensional features have component magnitudes near one, \(Z\) can itself be on the order of 64.

The timing is also ambiguous: a causal eligibility event should normally use a presynaptic state at \(t\) and a postsynaptic innovation or local sensitivity associated with the transition to \(t+1\), not a same-time raw state correlation.

**Required correction:** define a bounded local event

\[
z_{i,s}^t=\chi_i^tg_{i,s}^t\operatorname{clip}\!\left(
\frac{\langle \operatorname{LN}(p_i^t),\operatorname{LN}(\Delta s_j^{t+1})\rangle}{\sqrt{d_e}},-1,1
\right),
\]

then use a normalized exponential moving average

\[
e_{i,s}^{t+1}=\lambda_e e_{i,s}^t+(1-\lambda_e)z_{i,s}^t.
\]

If \(\lvert e_0\rvert\le1\), then \(\lvert e_t\rvert\le1\) for all \(t\).

### Equations (14)–(15): neuromodulation

**Verdict: CRITICAL.** A single scalar \(\mu^t\) broadcasts one learning sign to every eligible edge. In a large recurrent graph, that is extremely coarse credit assignment. It also creates a high-leverage poisoning surface.

**Required correction:** use a small number of bounded regional modulators and an independently computed trust gate:

\[
\mu_{i,s}^t=q_t\tanh(v_{r(i),r(j)}^\top m_t),
\qquad q_t\in[0,1].
\]

The trust gate must be produced by a protected monitor that the adapting network cannot directly rewrite.

### Equation (16): fast trace update

\[
f_{i,s}^{t+1}=\lambda_ff_{i,s}^t+\eta_f\mu^te_{i,s}^t.
\]

**Verdict: BLOCKER unless the inputs are bounded.** Combining the original Equations (13) and (16), if \(\lvert\mu_t\rvert\le M\) and \(\lvert z_t\rvert\le Z\), then

\[
\lvert f_t\rvert\lesssim
\frac{\eta_f MZ}{(1-\lambda_e)(1-\lambda_f)}.
\]

With \(\lambda_e=0.95\), \(\lambda_f=0.90\), and \(\eta_f=0.10\), the amplification coefficient is 20 before multiplying by \(MZ\).

A bounded replacement is

\[
f_{i,s}^{t+1}=\operatorname{clip}\!\left[
\lambda_ff_{i,s}^t+(1-\lambda_f)f_{\max}
\tanh(\eta_f\alpha_{i,s}\mu_{i,s}^te_{i,s}^{t+1}),
-f_{\max},f_{\max}
\right].
\]

### Equation (17): utility

\[
u_{i,s}^{t+1}=\lambda_uu_{i,s}^t+(1-\lambda_u)\lvert\mu^te_{i,s}^t\rvert.
\]

**Verdict: CRITICAL.** This measures update magnitude, not beneficial contribution. A harmful, noisy, adversarial, or chronically surprised edge can receive high utility.

**Required correction:** separate at least three concepts:

- **importance**: sensitivity of validated loss to the edge,
- **usage**: how often the edge contributes,
- **plastic value**: whether its update improved a trusted objective.

A sleep-time importance proxy is

\[
I_{i,s}\leftarrow \lambda_I I_{i,s}+(1-\lambda_I)
\operatorname{clip}\left(\left|c_{i,s}\frac{\partial\mathcal L_{\mathrm{anchor}}}{\partial c_{i,s}}\right|,0,I_{\max}\right).
\]

A cheaper wake proxy may be used, but pruning must be validated against replay or an anchor set.

### Equation (18): static metaplasticity

\[
\alpha=\operatorname{softplus}(\beta).
\]

**Verdict: CRITICAL.** Softplus is nonnegative but unbounded. Use \(\alpha_{\max}\sigma(\beta)\). Per-edge \(\alpha\) also adds 200 GB at 100B edges in FP16.

### Equation (19): dynamic metaplasticity

**Verdict: CRITICAL semantic inversion.** The equation drives \(\alpha\) toward \(\kappa\hat u\); high-utility edges become **more** plastic. If utility denotes consolidated importance, this increases overwrite risk exactly where stability is most needed.

**Required correction:** distinguish uncertainty from importance. One example is

\[
\alpha_{i,s}^{\star}=\alpha_{\max}
\frac{U^{\mathrm{uncert}}_{i,s}}
{U^{\mathrm{uncert}}_{i,s}+\lambda_I I_{i,s}+\varepsilon},
\]

followed by slow bounded tracking of \(\alpha^\star\).

### Equations (20)–(21): consolidation and decay

Original:

\[
w' = w+\eta_w\gamma(f),\qquad f'=\rho_ff.
\]

**Verdict: BLOCKER.** Before sleep,

\[
w_{\mathrm{eff}}=w+\alpha f.
\]

After sleep,

\[
w_{\mathrm{eff}}'=w+\eta_w\gamma(f)+\alpha\rho_ff.
\]

The behavior jump is

\[
\Delta w_{\mathrm{eff}}=\eta_w\gamma(f)-\alpha(1-\rho_f)f.
\]

This is zero only under a special equality that cannot generally hold with a global \(\eta_w\), nonlinear \(\gamma\), and edge-specific \(\alpha\).

**Function-preserving correction:** choose a transfer fraction \(q_{i,s}\in[0,1]\), freeze \(\alpha\) during transfer, and set

\[
\bar w_{i,s}=w_{i,s}+q_{i,s}\alpha_{i,s}f_{i,s},
\]

\[
\bar f_{i,s}=(1-q_{i,s})f_{i,s}.
\]

Then

\[
\bar w_{i,s}+\alpha_{i,s}\bar f_{i,s}=w_{i,s}+\alpha_{i,s}f_{i,s}.
\]

After this exact transfer, sleep may optimize the model under replay and trust-region constraints. The old model must be retained until validation passes.

### Equations (22)–(23): EWC/SI penalties

**Verdict: MAJOR to BLOCKER at target scale.** The equations are conventional, but v0.2 does not define how \(F\), \(\Omega\), or anchors are updated. Storing an importance and anchor per edge adds hundreds of gigabytes per field. Rewiring also changes slot semantics, invalidating an importance value tied only to slot index.

**Required correction:** use block/region importance, replay constraints, or sampled anchor sensitivities by default. If edge importance is retained, it must be keyed to semantic connection identity and reset when an edge is rewired.

---

## 3.3 Structural plasticity: Equations (24)–(28)

### Equation (24): schedule inequality

**Verdict: VALID WITH CONDITIONS.** Sleep-dominant rewiring is reasonable, but the paper needs explicit maximum structural change per transaction and a rollback policy.

### Equation (25): bottom-fraction pruning

**Verdict: CRITICAL.** For the current default \(k_{\mathrm{out}}=8\) and \(p_{\mathrm{prune}}=0.10\),

\[
\left\lfloor0.10\times8\right\rfloor=0,
\]

so rewiring never occurs. More generally, per-source forced bottom-fraction pruning removes edges even when all are useful and repeatedly kills newborn edges initialized with low utility.

**Required correction:** use a global or regional budget, threshold/hysteresis, stochastic expected counts, minimum age, and a probation interval. No edge should be pruned solely because a fixed fraction must be met.

### Equation (26): coactivation matrix

\[
C_{i,j}=\mathbb E_{\mathrm{replay}}[p_i\cdot s_j].
\]

**Verdict: BLOCKER at scale.** Materializing or evaluating all \(N^2\) pairs is impossible for the proposed system. The score can also be negative and is not normalized for feature dimension or frequency.

**Required correction:** build a candidate set \(\mathcal C_i\) of size \(M\ll N\) from region-local neighbors, approximate nearest neighbors, two-hop structure, gradient proposals, and random exploration. Score only \(j\in\mathcal C_i\).

### Equation (27): global target softmax

**Verdict: BLOCKER at scale.** A softmax over all \(N\) targets for every regrown slot is \(O(N)\) per edge and has no duplicate, self-loop, in-degree, diversity, or connectivity constraint.

**Required correction:** sample over \(\mathcal C_i\) only and reject proposals that violate graph invariants.

### Equation (28): regrowth initialization

**Verdict: CRITICAL.** Random nonzero initialization causes a function jump and may quantize to zero. The paper correctly resets all edge state; the current implementation fails to reset the old slow weight.

**Required correction:** initialize new effective contribution at zero, warm the route gradually, reset all semantic state, and assign an age/probation field:

\[
w\leftarrow0,\quad f\leftarrow0,\quad e\leftarrow0,\quad I\leftarrow0,
\quad g_{\mathrm{warm}}\leftarrow0.
\]

A permanent sparse backbone should remain non-rewirable so that global communication and graph connectivity are never lost.

---

## 3.4 Memory systems: Equations (29)–(34)

### Equations (29)–(31): working memory

**Verdict: MAJOR.** The write/read MLP output dimension must be \(K\). More importantly:

- there is no no-write action,
- every write is a hard overwrite,
- no erase gate, usage policy, or retention mechanism is defined,
- \(K=4\)–7 vector slots is not a meaningful capacity claim by itself because each vector can encode arbitrarily many entangled features.

**Required correction:** define no-write, gated interpolation, slot age/usage, and a measured information or task-capacity bottleneck rather than relying on a biological item-count analogy.

### Equation (32): episodic tuple

**Verdict: MAJOR.** The tuple omits next state, terminal flags, temporal context, provenance, trust, uncertainty, model/encoder version, and a stable retrieval key. It also reuses \(r_t\), already used for cortical aggregation.

### Equation (33): retrieval

**Verdict: CRITICAL under encoder drift.** If stored embeddings were produced by an encoder that changes online, old keys and new queries no longer occupy the same space. Approximate nearest-neighbor retrieval silently degrades unless memories are re-embedded or aligned.

**Required correction:** use a frozen or slowly updated key encoder, store key-version metadata, and support version adapters or background re-indexing. Replay should preserve trajectories and sample by diversity, conflict, importance, trust, and recency—not similarity alone.

### Equation (34): supersession link

**Verdict: MAJOR.** The idea is sound but not operationally defined for continuous latent records. A contradiction detector, proposition identity, confidence, provenance, and temporal validity model are required.

Episodic memory must also have a finite budget and compaction policy. “External” does not mean infinite.

---

## 3.5 Wake–sleep, quantization, and complexity: Equations (35)–(41)

### Equation (35): synaptic load

\[
S^t=\frac1E\sum\lvert f\rvert.
\]

**Verdict: CRITICAL as a safety metric.** A global mean can remain small while a small region saturates. It also ignores \(\alpha\), slow-weight scale, receiver concentration, and quantization scale.

Use quantiles and relative effective perturbation, for example

\[
S_q^t=Q_q\left(
\frac{\lvert\alpha f\rvert}
{\operatorname{RMS}(w)_{\mathrm{block}}+\varepsilon}
\right).
\]

### Equation (36): hidden-state drift

**Verdict: CRITICAL.** Comparing current hidden state to a state \(\Delta\) steps ago confounds environmental change, action, and legitimate memory with parameter drift. It can trigger sleep merely because the agent moved.

Use fixed anchor probes, prediction KL, replay loss, representation discrepancy, parameter change, and Jacobian metrics.

### Equation (37): sleep trigger

**Verdict: CRITICAL.** A single-threshold OR rule has no hysteresis, refractory period, minimum useful sleep duration, or completion criterion. It can oscillate into repeated sleep. Surprise is tracked but unused.

Use high thresholds to enter sleep, lower thresholds to exit, a minimum and maximum wake interval, a compute budget, and a transactional acceptance test.

### Equations (38)–(39): global downscaling and budget

Original:

\[
w\leftarrow\kappa_ww,\qquad f\leftarrow\kappa_ff.
\]

**Verdict: BLOCKER for lifelong retention.** Repeated global scaling causes exponential erasure:

\[
w^{(n)}=\kappa_w^nw^{(0)}.
\]

For \(\kappa_w=0.99\), only about 36.6% remains after 100 sleep cycles and about \(4.3\times10^{-5}\) remains after 1000 cycles, absent compensating updates. A global mean absolute-weight budget also does not bound local recurrent gain.

**Required correction:** remove unconditional slow-weight downscaling. Use receiver normalization, per-block constraints, excess-only regularization, and replay-constrained optimization. Any homeostatic transformation must either preserve function or be validated and rolled back.

### Equation (40): low-precision slow weights

**Verdict: VALID FOR STORAGE, CRITICAL FOR ONLINE UPDATES.** The symmetric quantizer is conventional, but small continual updates disappear when they are below the quantization step. Direct INT4 updates need stochastic rounding, residual accumulation, a higher-precision hot tier, or a specialized quantization-aware update rule.

INT4 should be treated as a research target, not the initial online-training baseline. A practical hierarchy is:

- cold slow edges: INT4/INT8,
- hot active edges: INT8/FP16 plus residual,
- fast state and eligibility: compressed and allocated only to active/plastic edges,
- periodic blockwise requantization with stochastic rounding.

### Equation (41): compute complexity

Original claim:

\[
O(|A_t|k_{\mathrm{out}}d_m).
\]

**Verdict: BLOCKER as a total-cost claim.** This counts only idealized message multiplication. It omits:

- active-node selection,
- query/key computation and target gathers,
- all-node recurrent updates,
- eligibility/utility updates,
- pooling/readout,
- random scatter-adds,
- sleep replay,
- candidate construction and rewiring,
- host/device movement.

A more honest wake-step form is

\[
O(C_{\mathrm{route}}+|A_t|C_\Phi+|E_t^{\mathrm{active}}|(d_k+d_m)
+|R_t|C_\Psi+C_{\mathrm{plastic}}),
\]

where \(R_t\) is the set of active receivers. The architecture is scalable only if routing, state updates, and plasticity state access are themselves hierarchical and sparse.

---

## 4. Additional architectural problems not explicit in the numbered equations

### 4.1 Parameter-count analogy is ambiguous

The paper targets 86–100B **slow synaptic parameters** while citing an estimate of human neuron count. Nodes and synapses are different quantities. The design must explicitly choose \(N\), \(k_{\mathrm{out}}\), \(E=Nk_{\mathrm{out}}\), node width \(d\), and the number of slow scalars per edge.

A scalar graph edge is also not equivalent in expressivity to an arbitrary dense-model parameter. The shared transforms carry much of the computational capacity.

### 4.2 Symmetry and dead-routing risk

With shared node machinery, zero slow weights, zero fast traces, identical initial states, and globally broadcast input, many nodes begin permutation-equivalent. Norm top-k can repeatedly choose the same indices. Fixed node/region embeddings, typed modules, stochastic routing, and load balancing are needed to break symmetry deliberately.

### 4.3 Input is not sparsely injected

The present implementation broadcasts the same projected input to every node and updates every node's GRU. Target-scale sparse computation requires designated sensory interfaces and hierarchical input routing.

### 4.4 Online learning signal is underdefined

The paper lists supervised prediction error and RL reward but does not define what trustworthy online target is available in deployment. A self-modifying agent cannot assume dense labels. The v0.3 specification must separate:

- self-supervised prediction signals,
- environmental reward,
- verified human feedback,
- untrusted observations,
- safety-critical updates requiring delayed approval.

### 4.5 Sleep must be a protected transaction

The adapting model should not be able to suppress sleep pressure, mark poisoned memories as trusted, or overwrite its own rollback checkpoint. Sleep scheduling, trust scoring, validation, and rollback should be implemented in a protected control plane.

---

## 5. Target-scale memory lower bound

Let \(E\) be the number of edge slots. The v0.2 per-edge proposal includes a slow weight, target index, fast state, eligibility, utility, and plasticity coefficient.

| Field | Bytes/edge | 86B edges | 100B edges |
|---|---:|---:|---:|
| Slow weight, INT4 | 0.5 | 43 GB | 50 GB |
| Target index, uint32 | 4 | 344 GB | 400 GB |
| Fast trace, FP16 | 2 | 172 GB | 200 GB |
| Eligibility, FP16 | 2 | 172 GB | 200 GB |
| Utility, FP16 | 2 | 172 GB | 200 GB |
| Plasticity coefficient, FP16 | 2 | 172 GB | 200 GB |
| **Subtotal** | **12.5** | **1.075 TB** | **1.250 TB** |

This subtotal excludes node states, edge types, gating biases, quantization scales, block metadata, replay memory, model parameters for \(\Phi/\Psi\), activations, optimizer state, checkpoints, redundancy, and communication buffers.

The uint32 assumption itself requires \(N<2^{32}\). At \(E=100\)B and \(k_{\mathrm{out}}=8\), \(N=12.5\)B and at least 34 target bits are required. With \(d=64\) FP16 node states, those nodes alone require 1.6 TB. Raising \(k_{\mathrm{out}}\) reduces \(N\) but increases each routing softmax and local edge workload.

### Required scale correction

The only plausible path to 100B total slow slots is a **tiered sparse-memory system**:

1. Only a small hot working set resides on accelerators.
2. Fast state and eligibility are allocated lazily to recently active plastic edges rather than all edge slots.
3. Plasticity coefficients are shared by blocks/types/regions.
4. Utilities are quantized, sketched, or computed during sleep for candidate edges.
5. Target indices use structured regional offsets or compressed block layouts.
6. Cold slow edges live in host/disaggregated storage and are fetched by region.
7. Routing is block-sparse and hierarchical; no global scan is allowed per step.

The 86–100B number should remain a long-term storage target, not an initial resident model configuration.

---

## 6. Paper–implementation discrepancies and concrete bugs

### 6.1 The graph core is not meta-trained

`PlasticGraphCore.step` runs under `torch.no_grad()`. The input projection, message MLP, GRU, and query/key projections are randomly initialized in the toy experiment and never optimized. Therefore, the experiment does not test whether offline meta-training can learn useful online plasticity.

### 6.2 Sparse activation masks output after dense computation

The implementation computes message transforms for all nodes, target hidden tensors for all slots, key projections for all slots, all edge messages, all-node GRU updates, and all-edge eligibility updates. The active mask is applied after most of this work. Runtime and memory therefore scale approximately with all nodes and slots, not only the selected active graph.

### 6.3 The implementation omits \(\alpha\)

The code uses

\[
w_{\mathrm{eff}}=w_{\mathrm{slow}}+f_{\mathrm{fast}},
\]

not Equation (6).

### 6.4 Eligibility timing and masking differ

The update uses the post-step hidden state for both the source and target dot product, does not multiply by active-source or routing gates, and updates eligibility for every slot. Inactive edges can therefore accumulate utility.

### 6.5 Pooling differs

The paper pools the active set; the code averages every node.

### 6.6 Rewiring is disabled by integer flooring

With \(k_{\mathrm{out}}=8\) and a 10% prune fraction, the code computes zero pruned slots.

### 6.7 Rewired slow weights are not reset

The code changes the target and clears fast/eligibility/utility state but leaves `w_slow[i, slot]` intact. A learned scalar is silently transferred to an unrelated target. This is a semantic corruption bug and contradicts Equation (28).

### 6.8 Rewiring permits self-loops and duplicates

No graph invariant prevents repeated targets, self-connections, receiver overload, or disconnected components.

### 6.9 Sleep replay mixes unrelated samples into one recurrent stream

The implementation clears fast/eligibility state once, then repeatedly samples random batches without resetting or segmenting recurrent hidden state per trajectory. This can create replay-order artifacts unrelated to the stored tasks.

### 6.10 Existing committed toy results are approximately chance

The binary-task accuracy values fluctuate around 0.5 in both sleep and no-sleep runs. Sleep lowers measured fast-state load but does not demonstrate learning or retention. This is expected given the untrained random core and should be reported as an implementation smoke test, not evidence for CSGN.

---

## 7. Catastrophic blockers versus difficult research risks

## 7.1 Blockers that make v0.2 fail as written

1. Non-function-preserving sleep consolidation and fast decay.
2. Unconditional global slow-weight downscaling over repeated sleep cycles.
3. Full per-edge runtime state at 86–100B scale.
4. Global all-pairs or all-target rewiring computations.
5. Dense execution hidden behind a sparse-activation claim.
6. Unbounded trace dynamics and uncontrolled scalar broadcast modulation.
7. No protection against representation drift in latent episodic keys.
8. Current rewiring bug that preserves the old slow weight after target replacement.
9. No transactional validation or rollback for consolidation and topology changes.
10. No trusted boundary around online learning, making poisoning a direct long-term-memory attack.

## 7.2 Serious risks that do not prove impossibility

1. Local plasticity may remain too biased for long-horizon credit assignment.
2. Meta-training may suppress plasticity as the easiest stable solution.
3. Sparse recurrent dynamics may oversquash information or form poor attractors.
4. Rewiring may fail to outperform a well-designed fixed sparse graph.
5. Replay may be too expensive or insufficiently representative.
6. Structured sparsity may reduce theoretical flexibility but is required for hardware speed.
7. Online representation learning may underperform frozen or pretrained features.
8. Continual adaptation remains vulnerable to adversarial or merely nonstationary data.

These are empirical questions. They require controlled falsification, not architectural optimism.

---

## 8. Corrected v0.3 design requirements

A valid v0.3 should include all of the following:

1. **Hierarchical region-to-column routing** with no global top-k scan.
2. **Typed multi-head messages** so outgoing messages are not rank-one.
3. **Receiver-normalized aggregation** with an explicit norm bound.
4. **Bounded recurrent state dynamics** and monitored Jacobian/spectral envelope.
5. **Normalized eligibility EMAs** with a formal boundedness statement.
6. **Bounded regional modulation** multiplied by a protected trust gate.
7. **Separate uncertainty, importance, usage, and plastic value metrics.**
8. **Function-preserving fast-to-slow transfer.**
9. **Replay-constrained sleep optimization** with an old-model distillation term.
10. **Transactional checkpoint, validation, acceptance, or rollback.**
11. **Permanent connectivity backbone plus probationary plastic slots.**
12. **Candidate-set rewiring** with graph invariants and no all-pairs matrix.
13. **Stable episodic keys, version metadata, provenance, and finite compaction.**
14. **Hot/cold edge-state hierarchy** instead of dense fast state for every edge.
15. **INT8/FP16 online baseline; INT4 only after residual-aware validation.**
16. **Actual sparse kernels and measured wall-clock/memory benchmarks.**
17. **Security tests for poisoning, sleep suppression, and malicious replay.**
18. **Explicit target-scale tuple** \((N,k_{\mathrm{out}},E,d,H,p_{\mathrm{active}})\).

---

## 9. Falsification program

The next experiments should not begin with Minecraft. Minecraft adds perception, partial observability, exploration, action semantics, reward design, and systems failures simultaneously. First isolate the mechanisms.

### Gate A: bounded local dynamics

- Verify the analytical bounds on \(e\), \(f\), receiver aggregation, and hidden-state norms.
- Run adversarial sequences designed to maximize each state.
- Reject any configuration with unbounded growth, NaNs, topology collapse, or repeated sleep loops.

### Gate B: meta-learned fast adaptation

Compare, at matched parameter and compute budgets:

- static recurrent baseline,
- replay-only baseline,
- differentiable-plastic baseline,
- CSGN without sleep,
- CSGN with function-preserving sleep.

Use cue binding, reversal learning, delayed reward, and task-switch streams. The graph core must be differentiably meta-trained; random shared modules are not a valid test.

### Gate C: retention and sleep

Pre-register:

- adaptation regret,
- old-task retention,
- forward transfer,
- calibration,
- anchor loss before/after sleep,
- effective-weight discontinuity,
- sleep compute cost.

A sleep transaction must be rejected if held-out anchor loss or a safety invariant exceeds its tolerance.

### Gate D: topology value

At matched memory and compute, compare:

- fixed random sparse topology,
- trained fixed sparse topology,
- magnitude/gradient rewiring,
- proposed utility/candidate rewiring.

Rewiring is retained only if it improves performance or efficiency beyond confidence intervals and does not increase catastrophic forgetting.

### Gate E: representation and episodic stability

Measure retrieval recall as the encoder changes. Compare frozen keys, version adapters, and full re-embedding. Test sequence replay versus independent tuple replay.

### Gate F: poisoning and recovery

Inject mislabeled, adversarial, repeated, and reward-manipulating experiences. Measure bounded influence, quarantine effectiveness, rollback recovery, and whether malicious memories survive consolidation.

### Gate G: systems scaling

Report actual resident bytes per edge, active bytes per step, random-gather bandwidth, scatter efficiency, kernel occupancy, and wall-clock scaling. A theoretical sparse FLOP count is insufficient.

Only after these gates pass should the system move to MineStudio/Minecraft.

---

## 10. Bottom line

The CSGN research direction remains plausible, but v0.2 overstates mathematical closure and scale feasibility. The correct next step is not to scale the present implementation. It is to replace the unstable equations, make consolidation function preserving and transactional, compress plastic state, constrain topology, and produce a meta-trained small system that demonstrably beats matched baselines.

The most important conceptual change is this:

> **Wake plasticity may propose memory changes; sleep may commit them only after function-preserving transfer, replay validation, safety checks, and rollback eligibility.**

That converts sleep from an uncontrolled second learning phase into a verifiable memory-commit protocol.

---

## References added by this audit

1. Miconi, Clune, Stanley. *Differentiable plasticity: training plastic neural networks with backpropagation.* 2018.
2. Miconi et al. *Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity.* 2020.
3. Evci et al. *Rigging the Lottery: Making All Tickets Winners.* 2020.
4. Lasby et al. *Dynamic Sparse Training with Structured Sparsity.* ICLR 2024.
5. Prabhu et al. *Random Representations Outperform Online Continually Learned Representations.* NeurIPS 2024.
6. Yoo et al. *Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning.* ICML 2024.
7. Bechler-Speicher and Eliasof. *A General Recipe for Contractive Graph Neural Networks.* 2024.
8. Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* 2024.
9. Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025.
10. Knoblauch, Husain, Diethe. *Optimal Continual Learning has Perfect Memory and is NP-hard.* ICML 2020.
11. Gurbuz and Dovrolis. *NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks.* ICML 2022.
12. Cong et al. *Test-Time Poisoning Attacks Against Test-Time Adaptation Models.* 2023.
13. Li and Ditzler. *PACOL: Poisoning Attacks Against Continual Learners.* 2023.
14. Panferov et al. *QuEST: Stable Training of LLMs with 1-Bit Weights and Activations.* ICML 2025.
15. Kim, Kim, Sohn. *Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective.* ICML 2025.

**End of audit.**
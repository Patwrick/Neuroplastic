# Cortical Synaptic Graph Networks (CSGN)
## A bounded neuroplastic graph architecture with transactional wake–sleep consolidation and budgeted structural adaptation

**Status:** Revised theory and design specification (v0.3); not yet empirically validated  
**Date:** 2026-08-11  
**Supersedes:** `Docs/csgn_paper_v0_2.md` for proposed mathematics and scaling claims  
**Companion audit:** `Docs/csgn_math_feasibility_audit_v0_3.md`

---

## Abstract

This paper proposes **Cortical Synaptic Graph Networks (CSGN)**, a sparse recurrent architecture intended to support rapid online adaptation, longer-term continual learning, replay, and fixed-budget structural change. CSGN separates memory into a slowly changing synaptic graph, bounded fast synaptic state, working memory, and a finite episodic store. Wake-time updates use bounded three-factor plasticity on only the active edge set. Sleep is reformulated as a **protected transaction**: fast state is first transferred into slow state without changing the effective synaptic function; replay-constrained optimization and limited topology proposals are then evaluated against frozen anchor probes and safety invariants; the transaction is committed only if validation passes, otherwise it is rolled back.

Version 0.3 removes several unsupported assumptions from the earlier proposal. Sparse activation is no longer treated as sufficient for sparse execution; all routing, state updates, plastic-state access, and rewiring must be hierarchical and sparse. Full fast state is not allocated to every slow edge. Rewiring never uses an all-pairs coactivation matrix. Incoming recurrent gain is receiver-normalized. Eligibility, fast state, recurrent state, and consolidation transfer have explicit bounds or invariants. The proposed scale of 86–100 billion slow edge slots is retained only as a possible long-term **tiered storage target**, not as a resident first implementation or a claim of biological equivalence.

The central hypothesis is falsifiable: after matched meta-training and under the same memory and compute budgets, bounded online plasticity plus validated transactional sleep should improve adaptation–retention trade-offs relative to static, replay-only, and differentiable-plastic baselines. The architecture should be rejected or simplified if these gains do not survive controlled ablations.

---

## 1. Claims, non-claims, and design contract

### 1.1 Claims being tested

CSGN tests whether a system can combine:

1. fast local adaptation during operation;
2. slower replay-constrained consolidation;
3. budgeted structural rewiring;
4. finite working and episodic memory;
5. explicit stability, provenance, validation, and rollback mechanisms;
6. sparse execution whose measured cost scales with the active graph rather than the full stored graph.

### 1.2 Non-claims

Version 0.3 does **not** claim:

- that the architecture is already an artificial general intelligence;
- that 86–100B graph edges are equivalent to human neurons or synapses;
- that replay eliminates catastrophic forgetting;
- that local plasticity solves arbitrary long-horizon credit assignment;
- that external episodic memory is infinite;
- that INT4 slow weights are immediately safe for direct online updates;
- that the present toy implementation validates the theory.

### 1.3 Explicit scale tuple

Every experiment must report

\[
\mathcal S=(N,k_{\mathrm{out}},E,d,H,p_{\mathrm{active}},\rho_{\mathrm{hot}}),
\qquad E=Nk_{\mathrm{out}},
\tag{1}
\]

where:

- \(N\): number of graph nodes or columns;
- \(k_{\mathrm{out}}\): stored outgoing slots per node;
- \(E\): total slow edge slots;
- \(d\): node-state width;
- \(H\): number of shared message heads;
- \(p_{\mathrm{active}}\): active node fraction per recurrent step;
- \(\rho_{\mathrm{hot}}=E_{\mathrm{hot}}/E\): fraction of edges with resident fast plastic state.

Recommended development scales are:

| Stage | \(N\) | \(k_{\mathrm{out}}\) | \(E\) | \(d\) | Purpose |
|---|---:|---:|---:|---:|---|
| P0 | 4,096 | 32 | 131,072 | 128 | Mathematical and mechanism tests |
| P1 | 65,536 | 64 | 4,194,304 | 128 | Sparse-kernel and replay tests |
| P2 | 1,048,576 | 64 | 67,108,864 | 64 | Hierarchical systems prototype |
| Long-term | architecture-dependent | architecture-dependent | 86–100B cold slots | architecture-dependent | Distributed/tiered storage research target |

Progression is gated by evidence, not by parameter count.

---

## 2. Architecture overview

CSGN contains six interacting subsystems:

1. **Interface encoders and decoders** for perception and action;
2. **Hierarchical Cortical Graph Core (CGC)** with permanent and plastic edge slots;
3. **Bounded fast synaptic state** allocated only to a hot edge set;
4. **Working memory** with explicit no-write and gated replacement;
5. **Finite episodic memory** with stable versioned keys, provenance, and replay policies;
6. **Protected maintenance controller** for trust gating, sleep scheduling, validation, checkpointing, and rollback.

The protected controller is outside the self-modifying parameter set. The adaptive model cannot directly disable sleep, alter anchor labels, declare untrusted events trusted, delete rollback snapshots, or relax safety thresholds.

---

## 3. Graph organization and hierarchical activation

### 3.1 Regions and edge slots

Nodes are partitioned into \(R\) regions. Node \(i\) has state

\[
h_i^t\in[-1,1]^d,
\tag{2}
\]

and region identifier \(r(i)\). Each outgoing slot \((i,s)\) stores:

- target \(\tau(i,s)\);
- slow scalar weight \(w_{i,s}\);
- message-head identifier \(\kappa(i,s)\in\{1,\ldots,H\}\);
- structural class: permanent backbone or plastic slot;
- compact age and topology metadata.

Fast state is allocated only while an edge is in the hot set \(\mathcal H_t\).

### 3.2 Permanent communication backbone

Each node or region reserves \(k_{\mathrm{backbone}}\) non-rewirable slots. The backbone is constructed to preserve required reachability—at minimum strong connectivity between regions or a formally specified communication diameter. Structural plasticity operates only on the remaining \(k_{\mathrm{plastic}}\) slots:

\[
k_{\mathrm{out}}=k_{\mathrm{backbone}}+k_{\mathrm{plastic}}.
\tag{3}
\]

This prevents plastic rewiring from disconnecting the computational graph.

### 3.3 Hierarchical activation

A full scan of all \(N\) nodes is not allowed at target scale. Routing first selects a bounded set of regions and then a bounded set of nodes inside each selected region:

\[
\mathcal G_t=\operatorname{RouteRegions}(c_t,\{\bar h_r^t\}_{r=1}^{R}),
\qquad |\mathcal G_t|\le K_R,
\tag{4}
\]

\[
A_t=\bigcup_{r\in\mathcal G_t}
\operatorname{TopKLocal}\left(\{a_i^t:i\in r\},K_N(r)\right).
\tag{5}
\]

The local score is learned rather than being only a state norm:

\[
a_i^t=v_a^\top\operatorname{LN}(h_i^t)+v_c^\top c_t+b_i^{\mathrm{type}}+\xi_i^t,
\tag{6}
\]

where \(\xi_i^t\) is controlled exploration noise during meta-training. Capacity penalties and regional quotas prevent a persistent rich-get-richer routing collapse. At very large \(R\), region routing itself must be a tree, product-key, or approximate-search procedure with sublinear lookup.

Hard routing may use a straight-through or relaxed estimator during meta-training. Structural rewiring is handled as a validated outer-loop operation and need not be differentiable.

---

## 4. Typed message passing with bounded receiver gain

### 4.1 Shared message heads

A source must be able to send more than one collinear message. Each active node computes \(H\) bounded shared message heads:

\[
m_{i,q}^t=M_{\max}\tanh\!\left(\Phi_q(\operatorname{LN}(h_i^t),c_t)\right),
\qquad q=1,\ldots,H.
\tag{7}
\]

Therefore \(\lVert m_{i,q}^t\rVert_2\le M_{\max}\sqrt{d_m}\). Each edge selects one compact head identifier \(\kappa(i,s)\). Larger-rank edge transformations are optional ablations, not part of the minimal scalable design.

### 4.2 Outgoing route gates with a null route

For active source \(i\), slot logits are

\[
\ell_{i,s}^t=\frac{(W_Qh_i^t)^\top(W_Kh_{\tau(i,s)}^t)}{\sqrt{d_k}}
+b_{\kappa(i,s)}+b_{r(i),r(\tau(i,s))}.
\tag{8}
\]

A null route \(s=0\) is included:

\[
(g_{i,0}^t,g_{i,1}^t,\ldots,g_{i,k_{\mathrm{out}}}^t)
=\operatorname{softmax}(\ell_{i,0}^t,\ell_{i,1}^t,\ldots),
\tag{9}
\]

so

\[
0\le g_{i,s}^t\le1,
\qquad \sum_{s=1}^{k_{\mathrm{out}}}g_{i,s}^t\le1.
\tag{10}
\]

For execution, only the top \(k_g\ll k_{\mathrm{out}}\) non-null gates are retained and renormalized. Keys are cached or gathered only for active sources and their stored targets.

### 4.3 Effective synaptic strength

The fast-expression coefficient is bounded and normally shared by edge block or type:

\[
\alpha_b=\alpha_{\max}\sigma(\beta_b),
\qquad b=b(i,s).
\tag{11}
\]

For an edge in the hot set, its effective weight is

\[
\widetilde w_{i,s}^t=\operatorname{clip}
\left(w_{i,s}+\alpha_{b(i,s)}f_{i,s}^t,-w_{\max},w_{\max}\right).
\tag{12}
\]

For a cold edge, \(f_{i,s}^t=0\). New structural edges have a warm-up gate \(v_{i,s}^t\in[0,1]\). The transmitted scalar is

\[
c_{i,s}^t=\mathbf1[i\in A_t]\,g_{i,s}^t\,v_{i,s}^t\,\widetilde w_{i,s}^t.
\tag{13}
\]

### 4.4 Receiver-normalized aggregation

Fixed out-degree does not bound in-degree. Let \(\mathcal E_j^t\) be active edges targeting receiver \(j\). Define

\[
Z_j^t=\max\left(1,\sum_{(i,s)\in\mathcal E_j^t}|c_{i,s}^t|\right),
\tag{14}
\]

\[
r_j^t=\frac{1}{Z_j^t}
\sum_{(i,s)\in\mathcal E_j^t}
c_{i,s}^t\,m_{i,\kappa(i,s)}^t.
\tag{15}
\]

If \(\lVert m_{i,q}^t\rVert_2\le M\), then by the triangle inequality,

\[
\lVert r_j^t\rVert_2
\le \frac{M\sum|c_{i,s}^t|}{\max(1,\sum|c_{i,s}^t|)}
\le M.
\tag{16}
\]

Thus receiver input norm is bounded independently of in-degree. A separate bounded load feature

\[
L_j^t=\tanh\left(\sum_{(i,s)\in\mathcal E_j^t}|c_{i,s}^t|\right)
\tag{17}
\]

preserves information about total incoming activity.

---

## 5. Bounded recurrent state dynamics

Only active senders, active receivers, working-memory interfaces, and sensory interface nodes are updated. Let this set be \(U_t\). For \(j\in U_t\), define

\[
\widehat h_j^{t+1}=\tanh\!\left(
A_h\operatorname{LN}(h_j^t)+B_rr_j^t+B_xx_{j,t}+B_cc_t+B_LL_j^t
\right),
\tag{18}
\]

\[
z_j^t=\sigma\!\left(
C_hh_j^t+C_rr_j^t+C_xx_{j,t}+C_cc_t
\right),
\tag{19}
\]

\[
h_j^{t+1}=(1-z_j^t)\odot h_j^t+z_j^t\odot\widehat h_j^{t+1}.
\tag{20}
\]

If \(h_j^0\in[-1,1]^d\), Equation (20) preserves \(h_j^t\in[-1,1]^d\) because it is an elementwise convex combination of bounded vectors. Nodes outside \(U_t\) retain their state, with any time decay applied lazily from a timestamp rather than by a dense full-graph update.

Bounded state does not by itself prevent chaotic sensitivity. Shared recurrent operators are spectrally regularized, and the maintenance validator monitors Jacobian-norm or finite-time Lyapunov proxies on anchor trajectories. A transaction is rejected if recurrent sensitivity exceeds a declared operating envelope.

A query-conditioned readout avoids indiscriminate mean pooling:

\[
\bar h_t=\sum_{i\in A_t}\pi_i^t h_i^t,
\qquad
\pi_i^t=\operatorname{softmax}_{i\in A_t}(q_y^\top K_yh_i^t),
\tag{21}
\]

\[
y_t=\Omega(\bar h_t,\mathrm{WM}_t).
\tag{22}
\]

---

## 6. Bounded online plasticity

### 6.1 Causal local event

For active edge \((i,s)\) targeting \(j\), define normalized pre-synaptic features and post-synaptic innovation:

\[
p_i^t=P\operatorname{LN}(h_i^t),
\qquad
d_j^{t+1}=D\operatorname{LN}(h_j^{t+1}-h_j^t).
\tag{23}
\]

The bounded local event is

\[
\zeta_{i,s}^t=\mathbf1[i\in A_t]\,g_{i,s}^t v_{i,s}^t
\operatorname{clip}\!\left(
\frac{\langle p_i^t,d_j^{t+1}\rangle}{\sqrt{d_e}},-1,1
\right).
\tag{24}
\]

Hence \(|\zeta_{i,s}^t|\le1\).

### 6.2 Multi-timescale eligibility

For delayed consequences, use \(L\) eligibility timescales:

\[
\lambda_{e,\ell}(\Delta t)=\exp(-\Delta t/\tau_{e,\ell}),
\tag{25}
\]

\[
e_{i,s,\ell}^{t+1}=\lambda_{e,\ell}e_{i,s,\ell}^{t}
+(1-\lambda_{e,\ell})\zeta_{i,s}^t.
\tag{26}
\]

If \(|e_{i,s,\ell}^0|\le1\), then Equation (26), a convex combination of values in \([-1,1]\), guarantees

\[
|e_{i,s,\ell}^t|\le1\quad\forall t.
\tag{27}
\]

The effective eligibility is a bounded mixture

\[
e_{i,s}^t=\sum_{\ell=1}^{L}\omega_\ell e_{i,s,\ell}^t,
\qquad \omega_\ell\ge0,\quad\sum_\ell\omega_\ell=1.
\tag{28}
\]

Eligibility state is allocated only to active hot edges and expires after a configured inactivity horizon.

### 6.3 Trusted regional modulation

The online learning signal contains reward-prediction error, self-supervised prediction error, novelty, homeostatic variables, and uncertainty:

\[
m_t^{\mathrm{mod}}=[\delta_t^{\mathrm{RPE}},\delta_t^{\mathrm{pred}},n_t,u_t,\ldots].
\tag{29}
\]

A protected trust estimator supplies \(q_t\in[0,1]\). The edge modulation is regional and bounded:

\[
\mu_{i,s}^t=q_t\tanh\!\left(
(v_{r(i),r(j)}^{\mathrm{mod}})^\top m_t^{\mathrm{mod}}
\right),
\qquad |\mu_{i,s}^t|\le1.
\tag{30}
\]

A single global scalar modulator remains an ablation, not the default. Untrusted or anomalous experiences can update short-lived state while being blocked from slow consolidation.

### 6.4 Bounded fast-state update

The plasticity rate is bounded:

\[
0\le\eta_{b(i,s)}^t\le\eta_{\max}.
\tag{31}
\]

With elapsed-time decay \(\lambda_f(\Delta t)=\exp(-\Delta t/\tau_f)\), update

\[
f_{i,s}^{t+1}=\operatorname{clip}\!\left[
\lambda_f f_{i,s}^t
+(1-\lambda_f)f_{\max}
\tanh\!\left(\eta_{b(i,s)}^t\mu_{i,s}^t e_{i,s}^{t+1}\right),
-f_{\max},f_{\max}
\right].
\tag{32}
\]

Therefore \(|f_{i,s}^t|\le f_{\max}\) by construction. Per-step and per-region write budgets additionally constrain

\[
\sum_{(i,s)\in\mathcal H_t}|f_{i,s}^{t+1}-f_{i,s}^{t}|
\le B_{\mathrm{wake}}.
\tag{33}
\]

### 6.5 Importance, usage, uncertainty, and plastic value

Version 0.3 does not use update magnitude as a synonym for importance. It tracks distinct quantities:

- usage \(U_{i,s}\): how often an edge materially contributes;
- importance \(I_{i,s}\): sensitivity of trusted anchor loss to the edge;
- uncertainty \(Q_{i,s}\): uncertainty or novelty associated with the edge's representation;
- plastic value \(V_{i,s}\): validated improvement attributable to recent plastic changes.

A sleep-time importance estimator may use

\[
I_{i,s}\leftarrow\lambda_I I_{i,s}
+(1-\lambda_I)\operatorname{clip}\!\left(
\left|c_{i,s}\frac{\partial\mathcal L_{\mathrm{anchor}}}{\partial c_{i,s}}\right|,0,I_{\max}
\right).
\tag{34}
\]

At scale, importance and plasticity rates are blockwise, quantized, sampled, or reconstructed during maintenance rather than stored in full precision for every cold edge.

Metaplasticity protects important edges while allowing uncertain edges to remain adaptable:

\[
\eta_b^{\star}=\eta_{\max}\sigma(a_QQ_b-a_II_b+a_VV_b+b_\eta),
\tag{35}
\]

\[
\eta_b^{t+1}=\operatorname{clip}\left((1-\rho_\eta)\eta_b^t+\rho_\eta\eta_b^{\star},0,\eta_{\max}\right).
\tag{36}
\]

High importance alone therefore decreases, rather than increases, plasticity.

---

## 7. Working and episodic memory

### 7.1 Working memory

Working memory contains \(K\) key–value slots \((k_a^t,v_a^t)\). The write policy has \(K+1\) outcomes, including no-write:

\[
\pi_{\mathrm{write}}(a\mid x_t,\bar h_t),
\qquad a\in\{0,1,\ldots,K\}.
\tag{37}
\]

For selected slot \(a>0\), a bounded write gate \(z_a^t\in[0,1]\) performs interpolation rather than unconditional overwrite:

\[
v_a^{t+1}=(1-z_a^t)v_a^t+z_a^t\operatorname{Encode}_v(x_t,\bar h_t),
\tag{38}
\]

with an analogous key update. Read attention is

\[
\alpha_a^t=\operatorname{softmax}_a
\left(\operatorname{sim}(q_t^{\mathrm{WM}},k_a^t)+b_a^{\mathrm{age}}\right),
\qquad
c_t^{\mathrm{WM}}=\sum_{a=1}^{K}\alpha_a^tv_a^t.
\tag{39}
\]

Slot count alone is not treated as an information-capacity proof; working-memory capacity must be evaluated empirically under controlled probes.

### 7.2 Episodic records

A replayable record stores a trajectory segment or transition with at least

\[
\mathcal E_n=(k_n,o_{n:n+L},a_{n:n+L},r_{n:n+L},d_{n:n+L},
\text{provenance},\text{trust},\text{uncertainty},\text{model version},\text{time}).
\tag{40}
\]

Keys are produced by a frozen or slowly updated **key encoder** separate from the rapidly adapting content encoder. Every key is versioned. When the key encoder changes, the system must use version adapters or background re-embedding; otherwise old memories and new queries become geometrically incompatible.

The episodic store has a finite byte budget. Compaction combines reservoir sampling, trajectory diversity, conflict coverage, importance, trust, and recency. Synthetic replay is provenance-tagged and cannot be treated as independently verified ground truth.

Replay sampling is a mixture rather than similarity-only retrieval:

\[
P(n)=\lambda_dP_{\mathrm{div}}(n)+\lambda_cP_{\mathrm{conflict}}(n)
+\lambda_iP_{\mathrm{importance}}(n)+\lambda_rP_{\mathrm{recency}}(n),
\tag{41}
\]

with nonnegative coefficients summing to one.

---

## 8. Sleep as a protected transaction

### 8.1 Pressure metrics

A global mean fast-state magnitude can hide local saturation. CSGN uses robust relative load quantiles:

\[
S_q^t=Q_q\left(
\frac{|\alpha_{b(i,s)}f_{i,s}^t|}
{\operatorname{RMS}(w)_{b(i,s)}+\varepsilon}
\right).
\tag{42}
\]

Drift is measured on fixed anchor probes, not by comparing hidden states from unrelated environmental moments. Example metrics include output KL divergence, anchor prediction loss, representation discrepancy, calibration shift, graph load concentration, and recurrent sensitivity. Terms entering the pressure score are robustly normalized against running reference distributions.

A pressure score is

\[
P_t=\omega_SS_q^t+\omega_AA_t+\omega_CC_t+\omega_T\min(1,T_{\mathrm{awake}}/T_{\max}).
\tag{43}
\]

Hysteresis uses \(\theta_{\mathrm{enter}}>\theta_{\mathrm{exit}}\). The controller enforces minimum and maximum wake intervals, a sleep compute budget, and a refractory period. Maintenance may run asynchronously on a shadow copy; severe drift can freeze further plastic writes until validation completes.

### 8.2 Exact fast-to-slow transfer

Before sleep, the effective unquantized synaptic value is

\[
w_{i,s}^{\mathrm{eff}}=w_{i,s}+\alpha_bf_{i,s}.
\tag{44}
\]

Choose a transfer fraction \(q_{i,s}\in[0,1]\), freeze \(\alpha_b\), and set

\[
w_{i,s}^{(0)}=w_{i,s}+q_{i,s}\alpha_bf_{i,s},
\tag{45}
\]

\[
f_{i,s}^{(0)}=(1-q_{i,s})f_{i,s}.
\tag{46}
\]

Then

\[
w_{i,s}^{(0)}+\alpha_bf_{i,s}^{(0)}
=w_{i,s}+\alpha_bf_{i,s},
\tag{47}
\]

so the transfer is exactly function preserving in real arithmetic. Quantization is postponed or accompanied by a residual accumulator; any remaining quantization error is explicitly measured before commit.

A hot edge may be evicted only after its fast state has decayed below tolerance, has been deliberately discarded as reversible context, or has passed through a validated transfer transaction. Silent eviction of nonzero fast state is prohibited.

### 8.3 Replay-constrained sleep objective

Let \(\Theta_0\) and \(G_0\) denote the transferred pre-optimization parameters and topology, and let \(M_{\mathrm{old}}\) be the frozen pre-sleep model. A candidate model minimizes

\[
\begin{aligned}
\mathcal L_{\mathrm{sleep}}(\Theta,G)=
&\;\lambda_n\mathcal L_{\mathrm{new\ replay}}
+\lambda_o\mathcal L_{\mathrm{old\ replay}}\\
&+\beta_y\,\mathbb E_{x\sim\mathcal A}
D_{\mathrm{KL}}\!\left(p_{M_{\mathrm{old}}}(\cdot|x)\,\|\,p_{\Theta,G}(\cdot|x)\right)\\
&+\beta_h\mathcal D_{\mathrm{repr}}(M_{\mathrm{old}},M_{\Theta,G};\mathcal A)\\
&+\beta_I\sum_b I_b\lVert\Theta_b-\Theta_{0,b}\rVert_2^2
+\beta_s\mathcal R_{\mathrm{stability}}(\Theta,G)
+\beta_g\mathcal R_{\mathrm{graph}}(G),
\end{aligned}
\tag{48}
\]

where \(\mathcal A\) is a protected anchor set. Layerwise proximal or curvature-aware constraints are compatible implementations of the replay geometry.

For reinforcement-learning trajectories, arbitrary replay is off-policy. Sleep must use an off-policy-corrected objective, a value/prediction objective, or behavior/policy distillation. A naïve on-policy policy-gradient update over arbitrary stored trajectories is not valid.

### 8.4 Atomic validation and commit

A sleep transaction proceeds as follows:

1. create an immutable checkpoint of parameters, topology, key versions, and controller state;
2. perform the function-preserving transfer in Equations (45)–(47);
3. train a shadow candidate under Equation (48);
4. propose only a bounded number of topology changes;
5. evaluate held-out anchors, current-task probes, calibration, safety constraints, recurrent stability, graph invariants, and resource budgets;
6. atomically commit only if every hard invariant passes and the declared objective improves sufficiently;
7. otherwise roll back the entire transaction.

Typical hard acceptance constraints are

\[
\Delta\mathcal L_{\mathrm{anchor}}\le\varepsilon_A,
\quad
D_{\mathrm{KL}}^{\mathrm{anchor}}\le\varepsilon_{\mathrm{KL}},
\quad
\widehat\rho_{\mathrm{dyn}}\le\rho_{\max},
\quad
\text{GraphInvariant}(G)=1.
\tag{49}
\]

Sleep is therefore a verifiable memory-commit protocol, not an unconditional second learning phase.

---

## 9. Budgeted structural plasticity

### 9.1 Pruning eligibility

An edge is eligible for pruning only if:

- it is a plastic rather than backbone slot;
- its age exceeds \(a_{\min}\);
- usage and importance are below thresholds;
- its effective contribution is small on anchor trajectories;
- a sampled ablation estimate is below tolerance;
- removing it does not violate topology or receiver-capacity invariants.

No source is forced to prune a fixed fraction when all its edges are useful. A sleep transaction has a global or regional rewire budget \(B_{\mathrm{rewire}}\).

### 9.2 Candidate-set regrowth

No all-pairs \(N\times N\) coactivation matrix is constructed. For source \(i\), build a candidate set

\[
\mathcal C_i=\mathcal C_i^{\mathrm{regional}}
\cup\mathcal C_i^{\mathrm{ANN}}
\cup\mathcal C_i^{\mathrm{two\ hop}}
\cup\mathcal C_i^{\mathrm{gradient}}
\cup\mathcal C_i^{\mathrm{random}},
\qquad |\mathcal C_i|=M\ll N.
\tag{50}
\]

A candidate score may be

\[
S_{i,j}=\lambda_c\widehat C_{i,j}
+\lambda_gG_{i,j}
+\lambda_nN_{i,j}
-\lambda_l\log(1+d_{\mathrm{in}}(j))
-\lambda_dD_{i,j}^{\mathrm{redundancy}}.
\tag{51}
\]

Targets are sampled only from valid candidates:

\[
P(j\mid i)=
(1-\epsilon)\operatorname{softmax}_{j\in\mathcal C_i}(S_{i,j}/T)
+\epsilon\operatorname{Uniform}(\mathcal C_i),
\tag{52}
\]

subject to no forbidden self-loop, no duplicate target/head pair, receiver-capacity limits, regional diversity, and backbone connectivity.

### 9.3 Function-safe initialization

A regrown edge starts with zero effective contribution and no inherited semantic state:

\[
w\leftarrow0,\quad f\leftarrow0,\quad e\leftarrow0,
\quad I\leftarrow0,\quad U\leftarrow0,\quad V\leftarrow0,
\quad v\leftarrow0,\quad \mathrm{age}\leftarrow0.
\tag{53}
\]

Its warm-up gate increases gradually,

\[
v(n)=\min(1,n/T_{\mathrm{warm}}),
\tag{54}
\]

and it receives a probation period before it can be pruned again. The full topology proposal remains subject to the sleep acceptance tests.

---

## 10. Meta-training and lifelong operation

### 10.1 Staged training

CSGN is not expected to self-organize from a blank random graph in a complex environment. Training is staged:

1. **Static representation pretraining:** train interface encoders, recurrent machinery, router, readout, and a fixed sparse graph;
2. **Fast-plasticity meta-training:** enable bounded fast state without sleep or rewiring;
3. **Transactional consolidation:** add function-preserving transfer, replay constraints, and rollback;
4. **Structural adaptation:** add a small number of validated plastic slots;
5. **Long-horizon environment training:** only after controlled mechanism tests pass.

### 10.2 Meta-objective

Across a task distribution \(\mathcal T\), the outer objective is

\[
\min_\Theta\;
\mathbb E_{\tau\sim\mathcal T}
\left[
\mathcal L_{\tau}^{\mathrm{query}}
\left(\operatorname{Adapt}_{\mathrm{wake}}(\Theta,D_{\tau}^{\mathrm{support}})
\right)
+\lambda_R\mathcal R_{\tau}
\right].
\tag{55}
\]

The differentiable wake update is unrolled over bounded horizons with truncated backpropagation, checkpointing, or implicit/local-gradient methods. Plasticity must be regularized against the degenerate solutions of either disabling all plasticity or saturating it everywhere.

Hard topology changes are not unrolled through long histories. They are proposed and selected by outer-loop validation during maintenance.

### 10.3 Online signal hierarchy

Runtime updates distinguish:

- self-supervised errors available directly from observation;
- environmental rewards, possibly delayed and noisy;
- verified human or system feedback;
- untrusted observations;
- safety-critical information requiring delayed approval.

Trust and provenance determine which memory tier an event may influence. Fast adaptation can be reversible; slow consolidation requires stronger evidence.

---

## 11. Quantization and tiered state

### 11.1 Why full per-edge fast state is infeasible

The v0.2 edge layout implied at least:

| Field | Bytes per edge |
|---|---:|
| INT4 slow weight | 0.5 |
| uint32 target | 4.0 |
| FP16 fast state | 2.0 |
| FP16 eligibility | 2.0 |
| FP16 utility | 2.0 |
| FP16 plasticity coefficient | 2.0 |
| **Subtotal** | **12.5** |

At 86B edges this is approximately 1.075 TB; at 100B edges it is approximately 1.250 TB, before node states, activations, replay, optimizer state, checkpoints, and communication buffers. This is not a resident single-accelerator design.

### 11.2 Hot/cold hierarchy

Version 0.3 uses

\[
M_{\mathrm{cold}}\approx E(b_w+b_\tau+b_{\mathrm{type}}+b_{\mathrm{coldmeta}}),
\tag{56}
\]

\[
M_{\mathrm{hot}}\approx E_{\mathrm{hot}}
(b_f+Lb_e+b_{\mathrm{hotmeta}}),
\qquad E_{\mathrm{hot}}=\rho_{\mathrm{hot}}E.
\tag{57}
\]

Only active or recently active plastic edges receive fast state and eligibility. Plasticity rates are blockwise by default. Cold edges may live in host, distributed, or disaggregated memory and are fetched by region. Target indices should use structured regional offsets or block layouts where possible; their compression ratio is an implementation result, not an assumption.

### 11.3 Online quantized updates

Initial experiments use FP16/BF16 or INT8 slow state. INT4 is tested only after stable higher-precision baselines exist. Let \(\Delta w\) be an accepted slow update and \(r\) a higher-precision residual. Define

\[
u=w+\Delta w+r,
\qquad
\widehat w=Q_b(u),
\qquad
r\leftarrow u-\widehat w,
\qquad
w\leftarrow\widehat w.
\tag{58}
\]

Use stochastic rounding and blockwise scale updates. Saturation rate, quantization error, and post-transfer output discontinuity are measured. An edge is not returned to cold storage until these checks pass.

---

## 12. Honest complexity accounting

Let \(E_t^{\mathrm{active}}\) be the executed edge set and \(U_t\) the updated node set. A wake step costs approximately

\[
\begin{aligned}
C_{\mathrm{wake}}=
&\;C_{\mathrm{hierarchical\ route}}
+O(|A_t|C_\Phi)
+O(|E_t^{\mathrm{active}}|d_k)\\
&+O(|E_t^{\mathrm{active}}|d_m)
+O(|U_t|C_\Psi)
+O(|E_t^{\mathrm{plastic}}|Ld_e)
+C_{\mathrm{WM}}+C_{\mathrm{readout}}.
\end{aligned}
\tag{59}
\]

The architecture is operationally sparse only if:

- inactive node transforms are not computed;
- inactive edge keys and messages are not gathered;
- recurrent updates are not applied to all \(N\) nodes;
- plastic state is touched only for active hot edges;
- routing does not perform a full global scan;
- scatter/gather kernels achieve measured bandwidth and occupancy.

Sleep cost must be reported separately as replay examples, optimizer steps, topology candidates, bytes moved, and wall-clock time. Sparse FLOP counts alone are insufficient.

---

## 13. Invariants and catastrophic failure conditions

A valid CSGN implementation must continuously enforce:

1. **State bound:** \(h_i^t\in[-1,1]^d\);
2. **Eligibility bound:** \(|e_{i,s,\ell}^t|\le1\);
3. **Fast-state bound:** \(|f_{i,s}^t|\le f_{\max}\);
4. **Receiver-input bound:** Equation (16);
5. **Per-step and per-sleep write budgets**;
6. **Permanent graph reachability and receiver capacity**;
7. **No duplicate or semantically stale state after rewiring**;
8. **Version-compatible episodic retrieval**;
9. **Immutable anchor, trust, and rollback control plane**;
10. **Transactional validation before slow or structural commit**.

The following outcomes are considered architecture-level failures unless corrected:

- sleep repeatedly changes behavior before optimization begins;
- slow memory decays merely because sleep occurs;
- fast state must be resident for every cold edge;
- rewiring requires all-pairs scoring or a global target softmax;
- measured execution remains dense despite sparse activation labels;
- the modulator or trust estimator can be poisoned into unrestricted slow writes;
- topology changes can disconnect the graph or transfer an old weight to a new semantic target;
- episodic retrieval collapses as the encoder drifts;
- the model cannot beat matched fixed-sparse and replay-only baselines.

---

## 14. Falsification program

### 14.1 Gate A: mathematical boundedness

Use adversarial sequences that maximize each state. Verify Equations (16), (27), and (32), bounded node states, finite write budgets, absence of NaNs, and stable sleep hysteresis. Test high in-degree, repeated reward spikes, delayed feedback, and pathological routing ties.

### 14.2 Gate B: meta-learned fast adaptation

On cue binding, reversal learning, delayed association, and nonstationary bandits, compare at matched parameters and compute:

- static recurrent graph;
- static graph plus replay;
- differentiable-plastic dense or sparse baseline;
- CSGN fast plasticity without sleep;
- CSGN fast plasticity with transactional sleep.

Report adaptation regret, sample efficiency, calibration, fast-state saturation, and wall-clock cost.

### 14.3 Gate C: consolidation and retention

Before and after every sleep transaction, report:

- effective-weight discontinuity after transfer;
- output KL on anchor probes;
- old-task retention;
- new-task gain;
- representation drift;
- rejected transaction rate;
- sleep compute and memory cost.

Function-preserving transfer should produce only bounded numerical error before replay optimization.

### 14.4 Gate D: structural value

Compare under the same edge count, replay, and compute:

- fixed random sparse graph;
- trained fixed sparse graph;
- magnitude-based rewiring;
- gradient-based rewiring;
- CSGN candidate-set rewiring.

Rewiring is retained only if it improves the adaptation–retention or efficiency frontier beyond confidence intervals without causing topology instability.

### 14.5 Gate E: episodic-key stability

Measure retrieval recall and downstream performance as the content encoder changes. Compare frozen keys, version adapters, background re-embedding, transition replay, and trajectory replay.

### 14.6 Gate F: poisoning and recovery

Inject mislabeled, repeated, adversarial, reward-manipulating, and synthetic self-confirming experiences. Measure bounded influence, quarantine, slow-write suppression, rollback recovery, and persistence after sleep.

### 14.7 Gate G: systems scaling

Report resident bytes per cold and hot edge, routing lookup complexity, gather/scatter bandwidth, kernel occupancy, active-set size, communication volume, and wall-clock scaling. Advance to Minecraft or another complex environment only after Gates A–G pass at smaller scales.

---

## 15. Discussion

CSGN's novelty does not arise from any single component. Differentiable plasticity, neuromodulation, replay, sparse training, test-time memory, and consolidation all have precedents. The research contribution, if validated, would be the integration of these mechanisms under explicit mathematical bounds, a tiered state model, structural invariants, and a transactional commit protocol.

The hardest unresolved issue is credit assignment. Local eligibility with regional modulation is a biased estimator in a large recurrent system. Meta-training may learn useful local rules, may suppress plasticity to preserve stability, or may fail as horizons grow. No equation in this paper guarantees general continual learning. The architecture must therefore be judged by controlled adaptation–retention frontiers and matched baselines.

The second unresolved issue is systems efficiency. Irregular sparse recurrent graphs can be memory-bandwidth limited even when their arithmetic count is low. Structured regional layouts may be less flexible than arbitrary graphs but are likely necessary for practical acceleration.

The third unresolved issue is memory governance. A lifelong learner must forget, compress, revise, and reject information. Version 0.3 replaces the earlier implication of memory that never runs out with a finite-budget policy whose losses and trade-offs are measurable.

---

## 16. Conclusion

CSGN remains a plausible research direction, but only after replacing unbounded and globally expensive mechanisms with a bounded, hierarchical, and verifiable specification. Version 0.3 makes five changes central:

1. recurrent messages and plastic state are executed only on a hierarchical active graph;
2. receiver aggregation, eligibility, fast state, and recurrent state have explicit bounds;
3. fast-to-slow transfer preserves effective synaptic function;
4. sleep and rewiring are shadow transactions with validation and rollback;
5. 86–100B slow edges are treated as a distributed cold-memory target, not an initial resident model.

The decisive experiment is not whether a large CSGN can be made to run. It is whether a small, rigorously controlled CSGN produces a reproducible adaptation–retention advantage over simpler matched systems. Scale should follow that result.

---

## References

1. Miconi, T., Clune, J., and Stanley, K. O. *Differentiable Plasticity: Training Plastic Neural Networks with Backpropagation.* ICML, 2018.
2. Miconi, T. et al. *Backpropamine: Training Self-Modifying Neural Networks with Differentiable Neuromodulated Plasticity.* ICLR, 2020.
3. Kirkpatrick, J. et al. *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS, 2017.
4. Zenke, F., Poole, B., and Ganguli, S. *Continual Learning Through Synaptic Intelligence.* ICML, 2017.
5. Gerstner, W. et al. *Eligibility Traces and Plasticity on Behavioral Time Scales.* Frontiers in Neural Circuits, 2018.
6. Tononi, G. and Cirelli, C. *Sleep and Synaptic Homeostasis: A Hypothesis.* Brain Research Bulletin, 2003.
7. Tadros, T. et al. *Sleep-like Unsupervised Replay Reduces Catastrophic Forgetting in Artificial Neural Networks.* Nature Communications, 2022.
8. Evci, U. et al. *Rigging the Lottery: Making All Tickets Winners.* ICML, 2020.
9. Lasby, M. et al. *Dynamic Sparse Training with Structured Sparsity.* ICLR, 2024.
10. Knoblauch, J., Husain, H., and Diethe, T. *Optimal Continual Learning Has Perfect Memory and Is NP-hard.* ICML, 2020.
11. Gurbuz, M. B. and Dovrolis, C. *NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks.* ICML, 2022.
12. Yoo, J. et al. *Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning.* ICML, 2024.
13. Urettini, E. and Carta, A. *Online Curvature-Aware Replay: Leveraging Second-Order Information for Online Continual Learning.* ICML, 2025.
14. Sun, Y. et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* 2024.
15. Behrouz, A., Zhong, P., and Mirrokni, V. *Titans: Learning to Memorize at Test Time.* 2025.
16. Yue, W., Liu, B., and Stone, P. *t-DGR: A Trajectory-Based Deep Generative Replay Method for Continual Learning in Decision Making.* CoLLAs, 2025.
17. Cong, T. et al. *Test-Time Poisoning Attacks Against Test-Time Adaptation Models.* 2023.
18. Li, Y. and Ditzler, G. *PACOL: Poisoning Attacks Against Continual Learners.* 2023.
19. Hernandez-Garcia, J. F. et al. *Reinitializing Weights vs Units for Maintaining Plasticity in Neural Networks.* 2025.

**End of document.**
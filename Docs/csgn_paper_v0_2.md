# Cortical Synaptic Graph Networks (CSGN)
## A neuroplastic, self-modifying architecture for lifelong learning with wake–sleep consolidation and structural rewiring

**Status:** Theory / design paper (v0.2)  
**Scope:** Model architecture and learning mechanisms only (environment-agnostic).  
**Target scale:** ~86–100B **slow** synaptic parameters; **fast** plastic state is additional runtime state (not counted as “parameters”).

---

## Table of contents

1. [Abstract](#abstract)  
2. [Introduction](#introduction)  
3. [Background and inspirations](#background-and-inspirations)  
4. [Model overview](#model-overview)  
5. [Cortical Graph Core](#cortical-graph-core)  
6. [Neuroplasticity mechanisms](#neuroplasticity-mechanisms)  
7. [Structural plasticity: pruning and regrowth](#structural-plasticity-pruning-and-regrowth)  
8. [Memory systems](#memory-systems)  
9. [Wake–sleep cycle](#wake–sleep-cycle)  
10. [Training strategy](#training-strategy)  
11. [Quantization and numerical considerations](#quantization-and-numerical-considerations)  
12. [Expected capabilities](#expected-capabilities)  
13. [Pros, cons, and failure modes](#pros-cons-and-failure-modes)  
14. [Complexity, scaling, and implementation notes](#complexity-scaling-and-implementation-notes)  
15. [Algorithms](#algorithms)  
16. [Suggested experiments](#suggested-experiments)  
17. [Discussion and open problems](#discussion-and-open-problems)  
18. [References](#references)  

---

## Abstract

Most modern neural systems are trained offline and deployed with fixed parameters, limiting their ability to **adapt safely and continually** in nonstationary settings. We propose **Cortical Synaptic Graph Networks (CSGN)**, a neuro-inspired architecture designed for **lifelong learning** via (i) continuously modulated **synaptic plasticity**, (ii) fixed-budget **structural rewiring**, and (iii) a compulsory **wake–sleep cycle** that performs **repair, consolidation, and renormalization**. The core network is a sparse recurrent graph of fixed-size “columns” connected by a fixed number of synapse slots, whose **effective connectivity is dynamically gated** and whose strengths can change both rapidly (**fast traces**) and slowly (**consolidated weights**). The wake phase prioritizes fast, local learning (three-factor plasticity); the sleep phase performs replay-driven consolidation, stability control, and topology optimization, reducing drift and catastrophic forgetting. CSGN synthesizes ideas from differentiable plasticity and neuromodulation, continual learning consolidation, sleep-like replay, sparse coding, and dynamic sparse topology updates.

---

## Introduction

### Motivation: beyond static “knowledge imprints”

Many high-performing neural models behave as **static circuits** at inference time: they compute dynamic activations but typically do not update their weights online. This works well for fixed tasks, but it conflicts with a key requirement of general intelligence: the ability to **learn continuously** from experience while avoiding catastrophic forgetting and unsafe drift.

Biological brains appear to solve this with:
- **synaptic plasticity** (changes in connection strengths),
- **structural plasticity** (forming/pruning connections),
- **short-term synaptic dynamics** (rapid, temporary changes),
- and **sleep-like phases** that support repair and consolidation.

CSGN is a design proposal for an artificial system with similar high-level properties:
- **Fixed number of “neurons/columns”** (no neuron growth),
- **Fixed synapse-slot budget** (constant parameter count),
- But **dynamic effective connectivity** and **online plasticity**,
- With an explicit **wake–sleep** mechanism for stability and consolidation.

### Design principles

1. **Plasticity as first-class computation**  
   Learning occurs during operation, not solely during offline training.

2. **Two timescales**  
   Rapid “fast” changes store immediate experience; slower consolidation integrates validated structure.

3. **Stability is not optional**  
   Sleep/repair is mandatory and enforced by internal homeostasis measures.

4. **Topology is adaptive but budgeted**  
   Connections can change, but total connection slots are constant (prune/regrow).

5. **Sparse activation**  
   Only a small fraction of units are strongly active at once, keeping compute tractable and encouraging efficiency.

---

## Background and inspirations

CSGN is motivated by the convergence of multiple research directions:

- **Differentiable plasticity**: neural networks with plastic components trained via gradient descent to learn useful within-episode learning rules.  
- **Neuromodulated plasticity**: learning a “dopamine-like” modulatory signal that gates plastic updates.  
- **Continual learning consolidation**: protecting important weights to reduce forgetting.  
- **Sleep-like replay**: replaying experiences to stabilize older knowledge and integrate new learning.  
- **Dynamic sparse training**: pruning and regrowing connections under a fixed budget to improve efficiency and performance.  
- **Sparse coding**: energy/efficiency-driven representation strategies and implicit regularization.

CSGN does **not** require spiking neurons, but it borrows the key *functional* ingredients of synaptic/structural plasticity and homeostasis.

---

## Model overview

CSGN is meant to be a general agent core behind any perception/action stack. This paper focuses on the **plastic core**, its **memory**, and its **wake–sleep** learning cycle.

### Components

1. **Perception encoder** produces embeddings \(x_t\) from raw inputs. (Not specified here.)  
2. **Working memory (WM)** with \(K\) slots (\(K \in [4,7]\) typical).  
3. **Cortical Graph Core (CGC)**: fixed nodes (“columns”) with synapse slots and plasticity.  
4. **Episodic memory (EM)**: external store supporting replay and cue-based retrieval.  
5. **Neuromodulators + homeostasis**: computes modulatory signals \(m_t\) and sleep pressure.  
6. **Wake–sleep scheduler**: triggers sleep and allocates consolidation/rewiring compute.

### Core constraints

- **Fixed nodes:** \(N\) constant.  
- **Fixed synapse slots:** out-degree \(k_{\text{out}}\) constant → total slots \(E = N k_{\text{out}}\).  
- **Dynamic effective connectivity:** gating makes the active graph input-dependent each step.  
- **Two-timescale synapses:** each slot has slow weight \(w\) and fast trace \(f\).  
- **Sleep-dominant rewiring:** major pruning/regrowth in sleep, minimal in wake.

---

## Cortical Graph Core

### Notation

- Time steps: \(t = 0,1,2,\dots\)  
- Nodes/columns: \(i \in \{1,\dots,N\}\)  
- Column state: \(h_i^t \in \mathbb{R}^{d}\)  
- Outgoing synapse slots per node: \(s \in \{1,\dots,k_{\text{out}}\}\)  
- Target of slot: \(\tau(i,s) \in \{1,\dots,N\}\)

Each synapse slot \((i,s)\) stores:

- **Slow synaptic weight**: \(w_{i,s}\)  
- **Fast plastic trace** (state): \(f_{i,s}^t\)  
- **Plasticity coefficient**: \(\alpha_{i,s}^t \ge 0\)  
- **Eligibility trace** (state): \(e_{i,s}^t\)  
- **Utility score** (state): \(u_{i,s}^t \ge 0\)  
- Optional: synapse type ID / region ID for structured priors.

### Sparse activation (“who participates”)

Define an activity score \(a_i^t\), e.g.

\[\tag{1}
a_i^t = \|h_i^t\|_2
\]

Select an active set \(A_t\) of size \(|A_t| = \lceil p_{\text{act}}N \rceil\) by top-\(p_{\text{act}}\) activity.

Only nodes in \(A_t\) send messages, enforcing sparse compute.

### Shared message transform

Each active node produces a message vector:

\[\tag{2}
m_i^t = \Phi(h_i^t, c^t) \in \mathbb{R}^{d_m}
\]

where \(c^t\) is a context vector (from WM and/or perception). \(\Phi\) is a shared (small) network and represents “cellular machinery,” while synapse weights implement routing/strength.

### Dynamic connectivity via gating

We define a gating factor \(g_{i,s}^t \in [0,1]\) for each outgoing synapse slot. One implementation is local attention over outgoing targets:

\[\tag{3}
q_i^t = W_Q h_i^t,\qquad k_j^t = W_K h_j^t
\]

For slot \(s\) targeting \(j=\tau(i,s)\), compute:

\[\tag{4}
\ell_{i,s}^t = \frac{(q_i^t)^\top k_j^t}{\sqrt{d_k}} + b_{i,s}
\]

Normalize over outgoing slots:

\[\tag{5}
g_{i,s}^t = \frac{\exp(\ell_{i,s}^t)}{\sum_{s'=1}^{k_{\text{out}}}\exp(\ell_{i,s'}^t)}
\]

This is a **dynamic routing policy** constrained to fixed synapse slots.

### Effective synaptic strength (slow + fast)

\[\tag{6}
\tilde{w}_{i,s}^t = w_{i,s} + \alpha_{i,s}^t f_{i,s}^t
\]

Gated coupling:

\[\tag{7}
c_{i,s}^t = g_{i,s}^t \cdot \tilde{w}_{i,s}^t
\]

### Incoming aggregation

Each node \(j\) receives:

\[\tag{8}
r_j^t = \sum_{i \in A_t}\sum_{s:\,\tau(i,s)=j} c_{i,s}^t \, m_i^t
\]

### State update (recurrent)

\[\tag{9}
h_j^{t+1} = \Psi\!\left(h_j^t, r_j^t, x_t, \text{WM}^t\right)
\]

\(\Psi\) may be GRU/LSTM-style gating, a stable residual MLP update, or another recurrent operator.

### Readout

Pool a global cortical summary:

\[\tag{10}
\bar{h}^t = \text{Pool}\big(\{h_i^t\}_{i\in A_t}\big)
\]

Compute readouts (policy, value, prediction, etc.) via a small head:

\[\tag{11}
y^t = \Omega(\bar{h}^t, \text{WM}^t)
\]

---

## Neuroplasticity mechanisms

CSGN includes three interlocking plastic processes:

1. **Fast synaptic plasticity** \(f_{i,s}^t\) (wake)  
2. **Metaplasticity** \(\alpha_{i,s}^t\) (plasticity-of-plasticity)  
3. **Slow consolidation** into \(w_{i,s}\) (sleep)

### Pre/post traces

Define pre- and post-synaptic features:

\[\tag{12}
p_i^t = \phi_{\text{pre}}(h_i^t),\qquad s_j^t = \phi_{\text{post}}(h_j^t)
\]

\(\phi_{\text{pre}}\) and \(\phi_{\text{post}}\) can be simple projections or nonlinearities.

### Eligibility traces

For synapse slot \((i,s)\) targeting \(j=\tau(i,s)\):

\[\tag{13}
e_{i,s}^{t+1} = \lambda_e e_{i,s}^t + \big(p_i^t\big)^\top s_j^t
\]

where \(\lambda_e \in [0,1)\) controls decay.

### Neuromodulators (what to learn from)

Define a modulatory signal \(m^t\). For generality, let it be a vector:

\[\tag{14}
m^t \in \mathbb{R}^{d_m^{(mod)}}
\]

We can define a scalar effective modulation \(\mu^t\) or a synapse-wise modulation via projections. A simple scalar modulation example:

\[\tag{15}
\mu^t = \mathcal{M}(\delta^t, n^t, \text{homeo}^t, \text{drift}^t)
\]

- \(\delta^t\): reward prediction error (extrinsic)  
- \(n^t\): novelty/surprise (intrinsic)  
- \(\text{homeo}^t\): homeostatic urgency (fatigue, deficit signals)  
- \(\text{drift}^t\): stability risk (see sleep triggers)

### Fast trace update (wake)

\[\tag{16}
f_{i,s}^{t+1} = \lambda_f f_{i,s}^t + \eta_f \, \mu^t \, e_{i,s}^t
\]

with decay \(\lambda_f \in [0,1)\) and fast learning rate \(\eta_f\).

This implements a three-factor learning rule: correlation accumulates in eligibility; modulation gates whether it changes synapses.

### Utility update (for pruning and consolidation)

\[\tag{17}
u_{i,s}^{t+1} = \lambda_u u_{i,s}^t + (1-\lambda_u)\,\left|\mu^t e_{i,s}^t\right|
\]

Interpretation: synapses repeatedly involved in high-modulator learning become “valuable.”

### Metaplasticity (who stays plastic)

Option A (static learnable plasticity coefficient):

\[\tag{18}
\alpha_{i,s} = \text{softplus}(\beta_{i,s})
\]

Option B (homeostatic metaplasticity, slow dynamics):

\[\tag{19}
\alpha_{i,s}^{t+1} = \text{clip}\!\left(\alpha_{i,s}^t + \eta_\alpha (\kappa \hat{u}_{i,s}^t - \alpha_{i,s}^t),\, 0, \alpha_{\max}\right)
\]

where \(\hat{u}\) is normalized utility and \(\eta_\alpha \ll \eta_f\).

### Slow consolidation (sleep)

During sleep, consolidate fast traces into slow weights:

\[\tag{20}
w_{i,s} \leftarrow w_{i,s} + \eta_w \,\mathbb{E}_{\text{replay}}\big[\gamma(f_{i,s})\big]
\]

where \(\gamma(\cdot)\) is a bounded transform (e.g., \(\tanh\), clip, or learned).

After consolidation, partially decay fast traces:

\[\tag{21}
f_{i,s} \leftarrow \rho_f f_{i,s},\qquad \rho_f \in (0,1)
\]

This enforces two-timescale memory: fast writes must be “validated” via replay to become stable.

### Optional: consolidation regularizers (continual learning)

To reduce catastrophic forgetting during slow updates, add penalties that protect important synapses:

**EWC-like penalty**

\[\tag{22}
\mathcal{L}_{\text{EWC}} = \sum_{i,s} \frac{\lambda_{\text{EWC}}}{2}\,F_{i,s}\,(w_{i,s}-w^\star_{i,s})^2
\]

**SI-like penalty**

\[\tag{23}
\mathcal{L}_{\text{SI}} = \sum_{i,s} \lambda_{\text{SI}}\,\Omega_{i,s}\,(w_{i,s}-w^\star_{i,s})^2
\]

These are applied during sleep consolidation steps, not during wake plasticity.

---

## Structural plasticity: pruning and regrowth

CSGN allows topology change under a constant synapse-slot budget. The guiding idea is “exuberant wiring + pruning,” implemented as **prune/regrow** under fixed \(E\).

### Sleep-dominant rewiring schedule

Let \(p_{\text{rewire,wake}}\) be extremely small (near zero). Most rewiring occurs in sleep:

\[\tag{24}
p_{\text{rewire,sleep}} \gg p_{\text{rewire,wake}}
\]

### Pruning rule

For each source node \(i\), prune the bottom fraction \(p_{\text{prune}}\) of slots by utility:

\[\tag{25}
\mathcal{P}(i) = \text{BottomK}\left(\{u_{i,s}\}_{s=1}^{k_{\text{out}}}, \left\lfloor p_{\text{prune}}k_{\text{out}}\right\rfloor\right)
\]

### Candidate selection for regrowth

Define a co-activation score from replay:

\[\tag{26}
C_{i,j} = \mathbb{E}_{\text{replay}}[\,p_i^t \cdot s_j^t\,]
\]

Sample new targets with an exploration mixture:

\[\tag{27}
\Pr(\tau(i,s)=j) = (1-\epsilon)\,\text{softmax}(C_{i,\cdot})_j + \epsilon \cdot \frac{1}{N}
\]

### Regrowth initialization

For each pruned slot \((i,s)\), assign new target \(j_{\text{new}}\) and initialize:

\[\tag{28}
\tau(i,s)\leftarrow j_{\text{new}},\quad w_{i,s}\leftarrow \mathcal{N}(0,\sigma_w^2),\quad f_{i,s}\leftarrow 0,\quad e_{i,s}\leftarrow 0,\quad u_{i,s}\leftarrow u_0
\]

This preserves constant parameter count and constant degree while changing connectivity.

---

## Memory systems

### Working memory (WM): explicit K-slot bottleneck

Working memory holds \(K\) “items” as vectors \(w_k^t \in \mathbb{R}^{d_w}\). Write and read are controlled by a small gating policy.

Write selection:

\[\tag{29}
\pi_{\text{write}}(k \mid x_t, \bar{h}^t) = \text{softmax}(\text{MLP}_{\text{write}}([x_t;\bar{h}^t]))
\]

Write update (overwrite chosen slot \(k^\star\)):

\[\tag{30}
w_{k^\star}^{t+1} = \text{Encode}(x_t,\bar{h}^t)
\]

Read/broadcast context:

\[\tag{31}
c^t = \sum_{k=1}^K \alpha_k^t w_k^t,\quad \alpha^t=\text{softmax}(\text{MLP}_{\text{read}}([x_t;\bar{h}^t]))
\]

This bottleneck is deliberate: it forces prioritization and prevents “infinite working memory” behavior.

### Episodic memory (EM): replay substrate

Episodic memory stores tuples:

\[\tag{32}
E_t = (z_t, a_t, r_t, \mu_t, \text{tags}_t)
\]

where \(z_t\) is a compressed latent (from encoder + cortex summary).

Retrieval uses approximate nearest neighbor search:

\[\tag{33}
\text{Retrieve}(q) = \text{TopK}\_{E}\big(\text{sim}(q,z)\big)
\]

EM supports replay, planning, and revision without continuously rewriting slow synapses.

### Contradictions without erasure

Instead of overwriting, store version links:

\[\tag{34}
P_{\text{new}} \xrightarrow{\text{supersedes}} P_{\text{old}}
\]

This supports “not true anymore” tagging and more robust reasoning about change.

---

## Wake–sleep cycle

CSGN operationalizes sleep as a computational phase that must occur to prevent drift and enable consolidation.

### Internal physiology variables

Track scalar pressures:

- fatigue \(F^t\) (time awake, effort)  
- synaptic load \(S^t\) (magnitude of fast traces)  
- drift risk \(D^t\) (inconsistency / instability)  
- surprise \(E^t\) (prediction error)  

Examples:

\[\tag{35}
S^t = \frac{1}{E}\sum_{i,s}|f_{i,s}^t|
\]

\[\tag{36}
D^t = \mathbb{E}_i \|h_i^t - h_i^{t-\Delta}\|_2
\]

### Sleep trigger

Sleep is triggered when any pressure exceeds threshold:

\[\tag{37}
\text{Sleep}(t) = \mathbb{1}\big[S^t>\theta_S \;\lor\; D^t>\theta_D \;\lor\; F^t>\theta_F\big]
\]

### Sleep operations (repair loop)

During sleep, perform:

1. **Replay selection** from episodic memory  
2. **Replay rollouts** through cortex  
3. **Consolidation**: update \(w\) from \(f\) and replay stats  
4. **Homeostatic downscaling/renormalization**  
5. **Structural rewiring**: prune/regrow by utility  
6. **Trace decay/reset**

### Homeostatic downscaling

Downscale to keep synaptic strengths bounded:

\[\tag{38}
w_{i,s} \leftarrow \kappa_w w_{i,s},\qquad f_{i,s}\leftarrow \kappa_f f_{i,s}
\]

Choose \(\kappa_w,\kappa_f \in (0,1]\) to satisfy a global budget:

\[\tag{39}
\frac{1}{E}\sum_{i,s}|\tilde{w}_{i,s}| \le B
\]

This prevents runaway potentiation and helps maintain a stable operating regime.

---

## Training strategy

CSGN is intended to be trained in two stages:

1) **Meta-training (offline):** learn representations and learning dynamics that make online plasticity effective.  
2) **Lifelong operation (online):** apply wake plasticity and sleep consolidation/rewiring.

### Meta-training objective

Meta-train parameters \(\Theta\) of:
- message transforms \(\Phi\),
- recurrent dynamics \(\Psi\),
- gating parameters \(W_Q,W_K\),
- modulators \(\mathcal{M}\),
- and optionally metaplasticity \(\alpha\)

so that, across tasks/episodes, the online learning dynamics improve performance.

### Online objectives

Maintain a mixture of losses (task-dependent):

- Predictive/self-supervised loss:
  \[
  \mathcal{L}_{\text{pred}} = \text{dist}(\hat{x}_{t+1}, x_{t+1})
  \]
- RL objective (if acting):
  \[
  \nabla_\Theta J \approx \mathbb{E}[\nabla_\Theta \log\pi_\Theta(a_t|s_t)\,(R_t - V_\Theta(s_t))]
  \]
- Regularizers:
  - sparsity regularization (encourage sparse active sets),
  - drift regularization,
  - replay reconstruction.

The modulatory signal \(\mu^t\) can incorporate prediction errors and reward prediction errors.

---

## Quantization and numerical considerations

### Low-precision slow synapses

Store \(w\) in \(b\)-bit quantized form:
\[\tag{40}
w_{i,s} = s_w \cdot q_{i,s},\qquad q_{i,s}\in\{-2^{b-1},\dots,2^{b-1}-1\}
\]

where \(b \in \{4,8\}\) and \(s_w\) is a scaling factor (per-tensor, per-row, or per-block).

### Mixed precision recommendation

- Slow weights \(w\): INT4/INT8  
- Fast traces \(f\) and eligibility \(e\): FP16/INT8 (state)  
- Activations and accumulations: FP16/BF16  
- Consolidation optimizer: higher precision where required

**Reason:** online updates can be numerically sensitive, so traces may need higher precision than stored slow weights.

---

## Expected capabilities

1. **Rapid within-episode learning**  
   Fast traces enable binding new associations quickly.

2. **Reduced catastrophic forgetting**  
   Replay + consolidation + optional EWC/SI-like stabilization reduce forgetting during sequential learning.

3. **Self-calibrated plasticity**  
   Metaplasticity allocates plasticity where uncertainty/novelty is high and reduces it in stabilized circuits.

4. **Dynamic connectivity**  
   Gating makes routing context-dependent, enabling flexible “on-the-fly” computation graphs.

5. **Long-term memory without “running out”**  
   Episodic memory scales externally; stable semantic structure accumulates slowly via consolidation.

---

## Pros, cons, and failure modes

### Pros
- **True online learning**: weights and connections can change during operation.  
- **Two-timescale stability**: fast learning is validated before consolidation.  
- **Structural adaptation**: efficiency improves via prune/regrow.  
- **Interpretability opportunities**: utilities and rewiring decisions are inspectable.  
- **Compute control**: sparse activation enables scaling under bounded compute.

### Cons / challenges
- **Stability–plasticity tuning** is delicate.  
- **Safety/poisoning risk**: online learning can be attacked by adversarial experience.  
- **Sparse acceleration** on hardware is hard (unstructured sparsity).  
- **Credit assignment**: local rules require meta-training to be effective.  
- **Evaluation**: must use rigorous continual learning ablations and drift measures.

### Failure modes
- **Runaway potentiation** (synapse saturation, loss of discrimination)  
- **Confabulation drift** (internal consistency breaks without sleep/repair)  
- **Over-pruning** (loss of capacity) / **under-pruning** (interference)  
- **Modulator collapse** (everything deemed important → chaotic plasticity)  
- **Topology thrashing** (too frequent rewiring)

Sleep triggers, homeostasis, clipping, and cautious consolidation mitigate these.

---

## Complexity, scaling, and implementation notes

### Compute per step
If only \(|A_t|\) nodes are active and each has \(k_{\text{out}}\) outgoing slots, message passing cost is approximately:

\[\tag{41}
\mathcal{O}(|A_t|\,k_{\text{out}}\,d_m)
\]

rather than dense \(\mathcal{O}(N^2)\) interactions.

### Memory footprint
- Slow synapses: \(E\) scalars (INT4/INT8)  
- Fast traces/eligibilities/utilities: \(E\) states (FP16/FP32)  
- Node states: \(N \times d\)  
- Episodic memory: external (disk/host RAM), scalable

### Practical note: structured sparsity
To leverage hardware efficiently, prefer:
- fixed out-degree \(k_{\text{out}}\),
- blockwise quantization,
- and batched index operations,
rather than irregular per-edge computation.

---

## Algorithms

### Algorithm 1: Wake step (online)

```text
Inputs: x_t, WM^t, h^t, topology τ, slow weights w, fast traces f, eligibility e, utilities u
Outputs: updated states h^{t+1}, WM^{t+1}, outputs y^t, updated f,e,u

1) WM^{t+1} ← WorkingMemoryUpdate(WM^t, x_t, h^t)
2) Determine active set A_t (top p_act by activity)
3) For i in A_t:
     m_i ← Φ(h_i^t, context(WM^{t+1}, x_t))
     compute gating g_{i,s} over s=1..k_out
     for each slot s:
        j ← τ(i,s)
        w_eff ← w_{i,s} + α_{i,s} * f_{i,s}
        send contribution: (g_{i,s} * w_eff) * m_i to j
4) Aggregate incoming messages r_j for each j
5) h^{t+1} ← Ψ(h^t, r, x_t, WM^{t+1})
6) Compute modulators μ^t = 𝓜(...)
7) Update eligibility: e ← λ_e e + pre(h^t)·post(h^t_targets)
8) Update fast traces: f ← λ_f f + η_f μ^t e
9) Update utility: u ← λ_u u + (1-λ_u) |μ^t e|
```

### Algorithm 2: Sleep cycle (repair, consolidation, rewiring)

```text
Inputs: episodic memory EM, topology τ, slow weights w, fast traces f, eligibility e, utilities u
Outputs: updated τ, w, f, e, u

1) Sample replay batches from EM (diverse + prioritized)
2) Run replay rollouts through cortex to compute consolidation stats
3) Homeostatic downscaling/renormalization of w and f
4) Consolidate: w ← w + η_w * E_replay[γ(f)]
5) Apply optional stability penalties (EWC/SI) during consolidation
6) Structural plasticity:
     For each node i:
        prune bottom p_prune fraction of slots by u_{i,s}
        regrow each pruned slot by sampling new targets (coactivation + exploration)
        reset f,e,u for regrown slots
7) Decay fast traces: f ← ρ_f f
8) Reset eligibility: e ← 0
```

---

## Suggested experiments

These experiments are **environment-agnostic** and test model mechanisms directly.

1) **Task-switching continual learning**
- Alternate tasks A/B/C.
- Measure retention of A after training on B.
- Ablations: no sleep, no replay, no consolidation, no rewiring.

2) **Within-episode rule shifts**
- Change mapping mid-episode (e.g., label permutation).
- Measure adaptation speed and recovery after sleep.

3) **Sleep deprivation drift**
- Disable sleep triggers.
- Measure rising prediction error variance and internal inconsistency proxies.

4) **Topology efficiency**
- Fixed topology vs sleep rewiring under same compute budget.
- Compare sample efficiency and final performance.

5) **Precision sensitivity**
- Compare INT4 vs INT8 slow weights for stability and long-horizon retention.

---

## Discussion and open problems

1) **Safety / poisoning**
- How to prevent adversarial experiences from shaping long-term weights?
- Potential mitigations: modulatory inhibition under suspicious signals, robust replay filters, quarantined learning.

2) **Stability metrics**
- What is the best measurable proxy for “hallucination-like drift” in a general agent core?
- Candidates: predictive calibration error, internal disagreement, oscillatory dynamics, memory inconsistency graphs.

3) **Hardware acceleration of sparsity**
- Unstructured sparsity is hard.
- Structured degree, block-sparse layouts, and batched indexing help.

4) **Learning rules**
- How expressive should the local plasticity be?
- Can we learn synapse-specific update functions \( \Delta f = \mathcal{U}(pre,post,mod) \)?

5) **Meta-training curricula**
- The meta-learning regime should gradually increase nonstationarity to train robust plasticity policies.

---

## References

The following references influenced the design and provide precedents for key components.

1. Miconi, Clune, Stanley. *Differentiable plasticity: training plastic neural networks with backpropagation.*  
   https://arxiv.org/abs/1804.02464

2. Miconi et al. *Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity.*  
   https://arxiv.org/abs/2002.10585

3. Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks (EWC).*  
   https://www.pnas.org/doi/10.1073/pnas.1611835114

4. Zenke, Poole, Ganguli. *Continual Learning Through Synaptic Intelligence (SI).*  
   https://proceedings.mlr.press/v70/zenke17a.html

5. Tononi & Cirelli. *Sleep and synaptic homeostasis hypothesis.*  
   https://pubmed.ncbi.nlm.nih.gov/14638388/

6. Tadros et al. *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks.*  
   https://www.nature.com/articles/s41467-022-34938-7

7. Evci et al. *Rigging the Lottery: Making All Tickets Winners (RigL).*  
   https://arxiv.org/abs/1911.11134

8. Cowan. *The magical number 4 in short-term memory.*  
   https://pubmed.ncbi.nlm.nih.gov/11515286/

9. Olshausen & Field. *Sparse coding of sensory inputs.*  
   https://pubmed.ncbi.nlm.nih.gov/15321069/

10. Azevedo et al. *Equal numbers of neuronal and nonneuronal cells… (≈86B neurons).*  
   https://pubmed.ncbi.nlm.nih.gov/19226510/

11. Gerstner et al. *Eligibility traces and plasticity on behavioral time scales.*  
   https://www.frontiersin.org/articles/10.3389/fncir.2018.00053/full

12. Pfeil et al. *Is a 4-bit synaptic weight resolution enough?*  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3398398/

---

**End of document.**

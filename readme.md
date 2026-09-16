# Neuroplastic

**Building toward AI that learns, remembers, and reorganizes through experience.**

Neuroplastic is an experimental research project exploring how an AI model can keep learning as it operates. The aim is to give a model the ability to adapt its connections, consolidate useful experiences into lasting memory, and allocate its capacity as the world around it changes.

The project centers on **Cortical Synaptic Graph Networks (CSGN)**: a proposed architecture that brings together recurrent graph computation, fast and slow synaptic memory, structural plasticity, and a wake–sleep learning cycle.

The central question is simple: **can a model learn something new quickly, retain what matters, and keep adapting within a fixed memory and compute budget?**

> **Status:** Early research and prototyping. This repository contains the architecture specification, a mathematical audit, and runnable experiments for individual mechanisms. Implementing and validating the complete CSGN v0.3 design is the next stage of the project.

## The model we want to build

Neuroplasticity provides the inspiration: experience can change both the strength of connections and how a network is organized. CSGN turns that idea into a set of mechanisms that can be implemented, measured, and challenged experimentally.

| Mechanism | Intended role in the model |
| --- | --- |
| **Fast plasticity** | Update short-lived synaptic state during interaction so recent feedback can immediately influence behavior. |
| **Slow synaptic memory** | Preserve useful structure and knowledge across experiences. |
| **Structural adaptation** | Reassign a limited number of connections toward useful relationships while preserving graph connectivity. |
| **Working and episodic memory** | Maintain immediate context and store selected experiences for later retrieval and replay, within explicit capacity limits. |
| **Wake–sleep consolidation** | Alternate online adaptation with maintenance periods that replay experience, evaluate changes, and commit accepted updates to longer-term memory. |
| **Sparse execution** | Concentrate computation on an active part of the stored graph, with measurable memory and compute costs. |

In the proposed wake phase, the model interacts with an environment and makes bounded local updates. During sleep, it transfers fast state into slower memory, replays selected experiences, and evaluates candidate changes to weights and connections. The v0.3 design requires validation and rollback before those changes are accepted.

The long-term goal is continual learning: acquiring new abilities, revising outdated associations, and retaining useful knowledge over time. Success depends on demonstrating that these mechanisms work together under controlled comparisons.

## What exists today

The code explores several parts of this design at small scales:

- **A neural plasticity benchmark:** a trainable neural policy with slow weights and reward-modulated fast state, evaluated on cue-to-action tasks. Plastic, static, and fast-only modes allow comparisons as cue mappings and frequencies change.
- **Connection-budget experiments:** cue agents compare fixed slot assignments with utility-based reassignment under the same slot budget.
- **An early graph-core prototype:** recurrent message passing, fast and slow edge state, eligibility traces, replay, and sleep/rewiring hooks.
- **Experiment tooling:** trajectory recording, checkpointing, seed sweeps, metrics, summaries, and plots, plus interchangeable toy and MineStudio environment backends.

The neural cue policy and the graph core are separate prototypes. The cue trainer currently uses supervised target labels to train slow parameters; evaluation adapts fast state using reward feedback. The graph core still has the implementation and mathematical limitations identified in the audit, and its committed toy runs do not establish a learning or retention advantage. The full v0.3 memory, sparse-execution, and transactional consolidation systems remain to be built and tested.

## Where Minecraft fits

Minecraft, through MineStudio, is one experimental environment for perception, action, and recorded experience. The architecture is intended to work across environments, and the core plasticity experiments run independently of Minecraft. Current MineStudio cue benchmarks use an added synthetic cue task to isolate adaptation.

Development starts with small tasks that isolate adaptation, forgetting, and connection allocation. Richer environments become useful once those mechanisms can be measured reliably.

## Try a small experiment

Use **Python 3.11 or 3.12** with the current dependency constraints. From the repository root, create and activate a virtual environment, then install the core and ML dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt -r requirements-ml.txt
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1` instead.

Check the toy environment:

```bash
python scripts/smoke_test_toy.py
```

Run a short CPU training example, then evaluate the saved policy on two seeds:

```bash
python scripts/train_plastic_cue.py --device cpu --episodes 5 --episode_len 20 --batch_size 8 --hidden_dim 32 --log_every 1 --save_path runs/plastic_cue_smoke.pt
python scripts/eval_plastic_cue.py --device cpu --checkpoint runs/plastic_cue_smoke.pt --tag QUICKSTART --steps 200 --seeds 0,1
```

This short run checks the training and evaluation pipeline. Assessing learning quality requires longer runs, multiple seeds, and matched baselines. Evaluation writes metrics and metadata under `data/rollouts/QUICKSTART/`.

See the [experiments and setup guide](Docs/experiments.md) for larger cue experiments, baseline comparisons, graph-core runs, GPU setup, plots, and optional MineStudio/Docker instructions.

## Research direction

The next milestone is a small, integrated CSGN implementation with reproducible evidence for its learning behavior. The work is organized around four questions:

1. **Adaptation:** Does online plasticity improve learning after a task changes, compared with matched static and replay-based models?
2. **Retention:** Can consolidation preserve earlier skills while incorporating new experience?
3. **Structure:** Does rewiring improve the use of a fixed connection budget compared with fixed topology?
4. **Stability and efficiency:** Are updates bounded, rejected changes recoverable, and execution costs consistent with the active graph?

Larger models and more complex environments should follow evidence from these tests.

## Design documents and code

- [CSGN v0.3 design paper](Docs/csgn_paper_v0_3.md) — the current proposed architecture and equations.
- [Mathematical and feasibility audit](Docs/csgn_math_feasibility_audit_v0_3.md) — the analysis of v0.2 that motivated the revised design, including current implementation gaps.
- [Residual risk register](Docs/csgn_v0_3_residual_risk_register.md) — unresolved questions about memory, concurrent updates, validation, and scaling.
- [Experiments and setup](Docs/experiments.md) — detailed commands and experiment recipes.
- [`csgn/model/`](csgn/model/) — graph-core and sleep-controller prototypes.
- [`csgn/models/`](csgn/models/) and [`csgn/agents/`](csgn/agents/) — neural policies and cue-agent baselines.
- [`csgn/envs/`](csgn/envs/) and [`csgn/env_backends/`](csgn/env_backends/) — cue tasks and environment adapters.
- [`scripts/`](scripts/) — training, evaluation, recording, and analysis tools.

The earlier [v0.2 paper](Docs/csgn_paper_v0_2.md) is retained for research history; v0.3 is the current design reference.

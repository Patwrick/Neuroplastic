# CSGN Minecraft Env Scaffold

Minimal backend-agnostic environment scaffold with:
- `toy` backend (runs without MineStudio)
- `minestudio` backend (lazy-imported MineStudio wrapper)

Python: `3.10+`

## Install core dependencies

```bash
pip install -r requirements-core.txt
```

## Run toy smoke test

```bash
python scripts/smoke_test_toy.py
```

## Optional: MineStudio backend

```bash
pip install -r requirements-minestudio.txt
python scripts/smoke_test_minestudio.py
```

If running headless, you may need `Xvfb` for MineStudio rendering.

## ML toy experiment (no MineStudio required)

```bash
pip install -r requirements-ml.txt
python scripts/run_toy_experiment.py
```

## GPU training (Blackwell/5090)

Install CUDA 12.8 PyTorch wheels on Windows:

```powershell
.\scripts\install_torch_cu128.ps1
```

If stable wheels do not yet support your setup, install nightly CUDA 12.8 wheels:

```powershell
.\scripts\install_torch_nightly_cu128.ps1
```

Verify CUDA availability:

```bash
python scripts/verify_torch_cuda.py
```

## Docker Desktop (Windows)

Run MineStudio smoke test in a Linux container:

```powershell
.\scripts\docker_minestudio_smoke.ps1
```

Open an interactive shell in the same Linux container setup:

```powershell
.\scripts\docker_minestudio_shell.ps1
```

On first run, MineStudio may prompt to download the simulator engine. Answer `Y` once; it is cached in the `minestudio_tmp` volume.

Note: Docker scripts use `docker/run_with_xvfb.sh` instead of `xvfb-run`, because `xvfb-run` can hang under Docker Desktop/WSL2.

## Recording trajectories (Docker)

Record a MineStudio rollout inside the Linux container:

```powershell
.\scripts\docker_record_minestudio_trajectory.ps1 -Steps 500 -Seed 0
```

Output is written to:

`data/trajectories/<run_id>/`

## Roll out BC policy in MineStudio (Docker)

```powershell
.\scripts\docker_rollout_bc.ps1 -Steps 200 -Seed 123
```

Output is written to:

`data/rollouts/<run_id>/`

## Neuroplastic cue benchmark (MineStudio Docker)

```powershell
.\scripts\docker_rollout_cue_plastic.ps1 -Steps 200 -Seed 0 -Mode plastic
.\scripts\docker_rollout_cue_plastic.ps1 -Steps 200 -Seed 0 -Mode static
```

Expected behavior: `plastic` should trend toward higher reward within an episode, especially when `change_every=0`.

Structural plasticity (fixed synapse-slot budget with online rewiring):

```powershell
.\scripts\docker_rollout_cue_plastic.ps1 -Steps 500 -Seed 0 -Mode slot_plastic -ExtraArgs "--K 32 --slots 8 --cue_dist uniform"
.\scripts\docker_rollout_cue_plastic.ps1 -Steps 500 -Seed 0 -Mode slot_plastic -ExtraArgs "--K 32 --slots 8 --cue_dist zipf --zipf_alpha 1.2"
```

The slot modes demonstrate structural plasticity: a limited number of connection slots rewires toward higher-utility cues. Under Zipf cue sampling, slot-limited agents should generally improve faster than with uniform sampling.

Fixed-slot hash baseline (same slot budget, no rewiring) for fair comparison:

```powershell
.\scripts\docker_cue_sweep.ps1 -Tag EXP_HASH_BASELINE -Steps 2000 -Seeds "0,1,2,3,4" -K 32 -CueDist zipf -ZipfAlpha 1.2 -Modes "slot_hash_plastic,slot_plastic" -SlotsList "8"
```

Toy smoke check for the hash baseline:

```bash
python scripts/rollout_cue_plastic.py --backend toy --steps 200 --seed 0 --mode slot_hash_plastic --K 8 --slots 4 --cue_dist uniform --out_dir data/rollouts/tmp_hash_smoke
```

## Summarize cue rollouts

```bash
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts --out_dir runs --dedupe longest
```

## Cue Sweep + Plots

Run a Docker sweep across seeds/modes and write rollouts under a tag folder:

```powershell
.\scripts\docker_cue_sweep.ps1 -Tag EXP6 -Steps 2000 -Seeds "0,1,2,3,4" -K 32 -CueDist zipf -ZipfAlpha 1.2 -SlotsList "8"
```

Experiment versioning: sweep runs stamp the current git commit and dirty flag into each run's `meta.json` (and per-step metrics columns), and summaries include these fields for reproducibility.

Example with explicit context shift/episode length:

```powershell
.\scripts\docker_cue_sweep.ps1 -Tag EXP7 -Steps 2000 -Seeds "0,1,2,3,4" -K 32 -CueDist zipf -ZipfAlpha 1.2 -SlotsList "8" -ChangeEvery 250 -EpisodeLen 500
```

Fair rewiring-vs-fixed mapping test under Zipf (permute cue IDs so popularity rank is decoupled from `cue_id % slots`):

```powershell
.\scripts\docker_cue_sweep.ps1 -Tag EXP9_ZIPF_PERMUTE -Steps 2000 -Seeds "0,1,2,3,4" -K 32 -CueDist zipf -ZipfAlpha 1.2 -Modes "slot_hash_plastic,slot_plastic,slot_hash_static,slot_static" -SlotsList "8" -ChangeEvery 250 -EpisodeLen 2000 -ExtraArgs "--permute_ids --permute_every 250 --permute_seed 123"
```

Summarize the tagged rollouts:

```bash
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts/EXP6 --out_dir runs/EXP6 --dedupe longest
```

For EXP7:

```bash
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts/EXP7 --out_dir runs/EXP7 --dedupe longest
```

For EXP9:

```bash
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts/EXP9_ZIPF_PERMUTE --out_dir runs/EXP9_ZIPF_PERMUTE --dedupe longest
```

Plot summary CSVs to PNG:

```bash
python scripts/plot_cue_summaries.py --summary_dir runs/EXP6
```

## Neural Plastic Cue Policy (No Minecraft)

Fast training on a pure-Python cue bandit environment:

```bash
python scripts/train_plastic_cue.py --device cuda --K 32 --cue_dist zipf --zipf_alpha 1.2 --permute_ids --episodes 2000
```

Evaluate to rollout-style metrics under a tag:

```bash
python scripts/eval_plastic_cue.py --checkpoint runs/plastic_cue.pt --tag EXP_NEURAL_PLASTIC --steps 2000 --seeds 0,1,2,3,4 --device cuda
```

EXP11 override example (checkpoint config + explicit eval overrides):

```bash
python scripts/eval_plastic_cue.py --device cuda --checkpoint runs/plastic_cue.pt --tag EXP11_NEURAL_PLASTIC --steps 2000 --seeds "0,1,2,3,4" --K 32 --cue_dist zipf --zipf_alpha 1.2 --permute_ids --permute_every 250
```

EXP12 paired neural baselines (plastic vs static) into one tag:

```bash
python scripts/train_plastic_cue.py --mode neural_plastic --device cuda --K 32 --cue_dist zipf --zipf_alpha 1.2 --permute_ids --save_path runs/plastic_cue_plastic.pt
python scripts/train_plastic_cue.py --mode neural_static --device cuda --K 32 --cue_dist zipf --zipf_alpha 1.2 --permute_ids --save_path runs/plastic_cue_static.pt
python scripts/eval_plastic_cue.py --device cuda --checkpoint runs/plastic_cue_plastic.pt --tag EXP12_NEURAL_BASELINES --steps 2000 --seeds "0,1,2,3,4" --permute_ids --permute_every 250
python scripts/eval_plastic_cue.py --device cuda --checkpoint runs/plastic_cue_static.pt --tag EXP12_NEURAL_BASELINES --steps 2000 --seeds "0,1,2,3,4" --permute_ids --permute_every 250
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts/EXP12_NEURAL_BASELINES --out_dir runs/EXP12_NEURAL_BASELINES --dedupe longest
python scripts/plot_cue_summaries.py --summary_dir runs/EXP12_NEURAL_BASELINES
```

Aggregate with existing summarizer:

```bash
python scripts/summarize_cue_rollouts.py --rollouts_dir data/rollouts/EXP_NEURAL_PLASTIC --out_dir runs/EXP_NEURAL_PLASTIC --dedupe longest
```

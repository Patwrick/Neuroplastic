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

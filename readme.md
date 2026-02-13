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

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

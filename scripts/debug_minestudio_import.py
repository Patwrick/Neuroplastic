from __future__ import annotations

import platform
import sys
import traceback


def main() -> None:
    numpy_exc: Exception | None = None
    np_version = "unknown"
    try:
        import numpy as np

        np_version = getattr(np, "__version__", "unknown")
    except Exception as exc:
        np_version = "<import failed>"
        numpy_exc = exc

    print("=== MineStudio Import Debug ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np_version}")

    if numpy_exc is not None:
        traceback.print_exception(numpy_exc)
        raise SystemExit(1)

    try:
        import minestudio

        if hasattr(minestudio, "__version__"):
            print(f"minestudio.__version__: {minestudio.__version__}")
        else:
            print("minestudio.__version__: <missing>")

        from minestudio.simulator import MinecraftSim

        print(f"MinecraftSim: {MinecraftSim}")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

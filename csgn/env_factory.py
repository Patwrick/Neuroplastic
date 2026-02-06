from __future__ import annotations

from csgn.env_api import EmbodiedEnv

ALLOWED_BACKENDS = ("toy", "minestudio")


def make_env(backend: str, **kwargs) -> EmbodiedEnv:
    key = backend.lower()

    if key == "toy":
        from csgn.env_backends.toy_backend import ToyEnv

        return ToyEnv(**kwargs)
    if key == "minestudio":
        from csgn.env_backends.minestudio_backend import MineStudioEnv

        return MineStudioEnv(**kwargs)

    allowed = ", ".join(ALLOWED_BACKENDS)
    raise ValueError(f"Unknown backend {backend!r}. Allowed backends: {allowed}")

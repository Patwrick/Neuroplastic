from __future__ import annotations

from typing import Any

import numpy as np


class CueBanditEnv:
    def __init__(
        self,
        K: int = 32,
        action_dim: int = 8,
        episode_steps: int = 200,
        cue_dist: str = "uniform",
        zipf_alpha: float = 1.2,
        permute_ids: bool = False,
        permute_every: int = 0,
        permute_seed: int | None = None,
        change_every: int = 0,
        batch_size: int = 1,
    ):
        self.K = int(K)
        self.action_dim = int(action_dim)
        self.episode_steps = int(episode_steps)
        self.cue_dist = str(cue_dist)
        self.zipf_alpha = float(zipf_alpha)
        self.permute_ids = bool(permute_ids)
        self.permute_every = int(permute_every)
        self.permute_seed = None if permute_seed is None else int(permute_seed)
        self.change_every = int(change_every)
        self.batch_size = int(batch_size)

        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self.cue_dist not in ("uniform", "zipf"):
            raise ValueError(f"cue_dist must be 'uniform' or 'zipf', got {cue_dist!r}")
        if self.permute_every < 0:
            raise ValueError(f"permute_every must be >= 0, got {permute_every}")
        if self.change_every < 0:
            raise ValueError(f"change_every must be >= 0, got {change_every}")

        self._base_seed = 0
        self._rngs: list[np.random.Generator] = []
        self._perm_rngs: list[np.random.Generator] = []
        self._mapping = np.zeros((self.batch_size, self.K), dtype=np.int64)
        self._perm = np.tile(np.arange(self.K, dtype=np.int64), (self.batch_size, 1))
        self._perm_idx = np.zeros((self.batch_size,), dtype=np.int64)
        self._cue_id = np.zeros((self.batch_size,), dtype=np.int64)
        self._t = np.zeros((self.batch_size,), dtype=np.int64)
        self._zipf_probs = self._build_zipf_probs()

        self._init_rngs(base_seed=0)

    def _build_zipf_probs(self) -> np.ndarray:
        idx = np.arange(1, self.K + 1, dtype=np.float64)
        weights = 1.0 / np.power(idx, self.zipf_alpha)
        return (weights / np.sum(weights)).astype(np.float64)

    def _init_rngs(self, base_seed: int) -> None:
        self._base_seed = int(base_seed)
        self._rngs = []
        self._perm_rngs = []
        for i in range(self.batch_size):
            seed_i = self._base_seed + 10007 * i
            perm_seed_i = (
                (self.permute_seed if self.permute_seed is not None else self._base_seed) + 10007 * i
            )
            self._rngs.append(np.random.default_rng(int(seed_i)))
            self._perm_rngs.append(np.random.default_rng(int(perm_seed_i)))

    def _sample_mapping_row(self, idx: int) -> None:
        self._mapping[idx] = self._rngs[idx].integers(0, self.action_dim, size=(self.K,), dtype=np.int64)

    def _refresh_perm_row(self, idx: int, increment: bool) -> None:
        if self.permute_ids:
            self._perm[idx] = self._perm_rngs[idx].permutation(self.K).astype(np.int64)
        else:
            self._perm[idx] = np.arange(self.K, dtype=np.int64)
        if increment:
            self._perm_idx[idx] += 1

    def _sample_rank_row(self, idx: int) -> int:
        if self.cue_dist == "uniform":
            return int(self._rngs[idx].integers(0, self.K))
        return int(self._rngs[idx].choice(self.K, p=self._zipf_probs))

    def _sample_cue_row(self, idx: int) -> int:
        rank = self._sample_rank_row(idx)
        if self.permute_ids:
            return int(self._perm[idx, rank])
        return int(rank)

    def _build_obs(self) -> dict[str, np.ndarray]:
        cue_onehot = np.zeros((self.batch_size, self.K), dtype=np.float32)
        cue_onehot[np.arange(self.batch_size), self._cue_id] = 1.0
        target = self._mapping[np.arange(self.batch_size), self._cue_id]
        return {
            "cue_id": self._cue_id.copy(),
            "cue_onehot": cue_onehot,
            "target_action_idx": target.copy().astype(np.int64),
        }

    def _build_info(self) -> dict[str, Any]:
        return {
            "cue_dist": self.cue_dist,
            "zipf_alpha": self.zipf_alpha,
            "permute_ids": bool(self.permute_ids),
            "permute_every": int(self.permute_every),
            "permute_seed": self.permute_seed,
            "permute_idx": self._perm_idx.copy(),
            "t_global": self._t.copy(),
        }

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._init_rngs(base_seed=int(seed) if seed is not None else 0)
        self._t[:] = 0
        self._perm_idx[:] = 0

        for i in range(self.batch_size):
            self._sample_mapping_row(i)
            self._refresh_perm_row(i, increment=False)
            self._cue_id[i] = self._sample_cue_row(i)

        return self._build_obs(), self._build_info()

    def step(
        self, action_idx: int | np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        action = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        if action.size == 1 and self.batch_size > 1:
            action = np.repeat(action, self.batch_size)
        if action.size != self.batch_size:
            raise ValueError(f"action_idx must have size {self.batch_size}, got {action.size}")

        target = self._mapping[np.arange(self.batch_size), self._cue_id]
        reward = (action == target).astype(np.float32)
        terminated = np.zeros((self.batch_size,), dtype=bool)

        self._t += 1

        if self.change_every > 0:
            change_mask = (self._t % self.change_every) == 0
            for i in np.where(change_mask)[0]:
                self._sample_mapping_row(int(i))

        if self.permute_ids and self.permute_every > 0:
            perm_mask = (self._t > 0) & ((self._t % self.permute_every) == 0)
            for i in np.where(perm_mask)[0]:
                self._refresh_perm_row(int(i), increment=True)

        for i in range(self.batch_size):
            self._cue_id[i] = self._sample_cue_row(i)

        if self.episode_steps > 0:
            truncated = self._t >= self.episode_steps
        else:
            truncated = np.zeros((self.batch_size,), dtype=bool)

        obs = self._build_obs()
        info = self._build_info()
        info["target_action_idx"] = target.copy().astype(np.int64)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        # Included for interface symmetry with other envs.
        return

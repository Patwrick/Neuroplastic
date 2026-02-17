from __future__ import annotations

from typing import Any

import numpy as np

from csgn.env_api import EmbodiedEnv, Obs, StepReturn


class CueRewardWrapper(EmbodiedEnv):
    def __init__(
        self,
        env: EmbodiedEnv,
        K: int = 2,
        episode_steps: int = 200,
        change_every: int = 0,
        cue_dist: str = "uniform",
        zipf_alpha: float = 1.2,
        permute_ids: bool = False,
        permute_every: int = 0,
        permute_seed: int | None = None,
    ):
        self._env = env
        self._K = int(K)
        self._episode_steps = int(episode_steps)
        self._change_every = int(change_every)
        self._cue_dist = str(cue_dist)
        self._zipf_alpha = float(zipf_alpha)
        self._permute_ids = bool(permute_ids)
        self._permute_every = int(permute_every)
        self._permute_seed = None if permute_seed is None else int(permute_seed)

        if self._K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if self._cue_dist not in ("uniform", "zipf"):
            raise ValueError(f"cue_dist must be 'uniform' or 'zipf', got {cue_dist!r}")
        if self._permute_every < 0:
            raise ValueError(f"permute_every must be >= 0, got {permute_every}")

        self._rng = np.random.default_rng(0)
        self._perm_rng = np.random.default_rng(0)
        self._mapping = np.zeros(self._K, dtype=np.int64)
        self._perm = np.arange(self._K, dtype=np.int64)
        self._perm_idx = 0
        self._cue_id = 0
        self._t = 0
        self._zipf_probs = self._build_zipf_probs()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    def _sample_mapping(self) -> None:
        self._mapping = self._rng.integers(0, 2, size=(self._K,), dtype=np.int64)

    def _build_zipf_probs(self) -> np.ndarray:
        idx = np.arange(1, self._K + 1, dtype=np.float64)
        weights = 1.0 / np.power(idx, self._zipf_alpha)
        probs = weights / np.sum(weights)
        return probs.astype(np.float64)

    def _refresh_permutation(self, increment: bool) -> None:
        if self._permute_ids:
            self._perm = self._perm_rng.permutation(self._K).astype(np.int64)
        else:
            self._perm = np.arange(self._K, dtype=np.int64)
        if increment:
            self._perm_idx += 1

    def _sample_rank(self) -> int:
        if self._cue_dist == "uniform":
            return int(self._rng.integers(0, self._K))
        return int(self._rng.choice(self._K, p=self._zipf_probs))

    def _sample_cue(self) -> int:
        rank = self._sample_rank()
        if self._permute_ids:
            return int(self._perm[rank])
        return int(rank)

    def _ensure_obs_dict(self, obs: Any) -> Obs:
        if isinstance(obs, dict):
            return obs
        return {"raw": {"obs": obs}}

    def _inject_cue(self, obs: Any, cue_id: int) -> Obs:
        out = self._ensure_obs_dict(obs)
        onehot = np.zeros((self._K,), dtype=np.float32)
        onehot[int(cue_id)] = 1.0
        out["cue_id"] = int(cue_id)
        out["cue_onehot"] = onehot
        out["cue_target"] = int(self._mapping[int(cue_id)])
        return out

    def _chosen_action_id(self, action: Any) -> int:
        action_dict = action if isinstance(action, dict) else {}
        forward_pressed = float(action_dict.get("forward", 0)) > 0.5
        back_pressed = float(action_dict.get("back", 0)) > 0.5
        if forward_pressed and not back_pressed:
            return 0
        if back_pressed and not forward_pressed:
            return 1
        return 0

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict]:
        obs, info = self._env.reset(seed=seed)

        self._t = 0
        self._rng = np.random.default_rng(int(seed) if seed is not None else 0)
        perm_seed_value = self._permute_seed if self._permute_seed is not None else (int(seed) if seed is not None else 0)
        self._perm_rng = np.random.default_rng(int(perm_seed_value))
        self._perm_idx = 0
        self._refresh_permutation(increment=False)
        self._sample_mapping()
        self._cue_id = self._sample_cue()

        wrapped_obs = self._inject_cue(obs, self._cue_id)
        out_info = dict(info) if isinstance(info, dict) else {"raw_info": info}
        out_info["cue_id"] = int(self._cue_id)
        out_info["cue_target"] = int(self._mapping[self._cue_id])
        out_info["cue_K"] = int(self._K)
        out_info["cue_dist"] = self._cue_dist
        out_info["zipf_alpha"] = self._zipf_alpha
        out_info["permute_ids"] = bool(self._permute_ids)
        out_info["permute_every"] = int(self._permute_every)
        out_info["permute_seed"] = self._permute_seed
        out_info["permute_idx"] = int(self._perm_idx)
        return wrapped_obs, out_info

    def step(self, action: Any) -> StepReturn:
        chosen = self._chosen_action_id(action)
        target = int(self._mapping[self._cue_id])
        synthetic_reward = 1.0 if chosen == target else 0.0

        obs, _reward, terminated, truncated, info = self._env.step(action)

        self._t += 1
        if self._change_every > 0 and self._t % self._change_every == 0:
            self._sample_mapping()
        if self._permute_ids and self._permute_every > 0 and self._t > 0 and self._t % self._permute_every == 0:
            self._refresh_permutation(increment=True)

        self._cue_id = self._sample_cue()

        wrapped_obs = self._inject_cue(obs, self._cue_id)
        out_info = dict(info) if isinstance(info, dict) else {"raw_info": info}
        out_info["cue_id"] = int(self._cue_id)
        out_info["cue_target"] = int(self._mapping[self._cue_id])
        out_info["synthetic_reward"] = float(synthetic_reward)
        out_info["t"] = int(self._t)
        out_info["cue_dist"] = self._cue_dist
        out_info["zipf_alpha"] = self._zipf_alpha
        out_info["permute_ids"] = bool(self._permute_ids)
        out_info["permute_every"] = int(self._permute_every)
        out_info["permute_seed"] = self._permute_seed
        out_info["permute_idx"] = int(self._perm_idx)

        if self._episode_steps > 0 and self._t >= self._episode_steps:
            truncated = True

        return wrapped_obs, float(synthetic_reward), bool(terminated), bool(truncated), out_info

    def close(self) -> None:
        self._env.close()

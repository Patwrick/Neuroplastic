from __future__ import annotations

import torch


class TaskSwitchStream:
    def __init__(self, d_in: int = 16, seed: int = 0):
        if d_in <= 0:
            raise ValueError(f"d_in must be > 0, got {d_in!r}")

        self.d_in = int(d_in)
        self._train_gen = torch.Generator(device="cpu")
        self._train_gen.manual_seed(seed)
        self._eval_gen = torch.Generator(device="cpu")
        self._eval_gen.manual_seed(seed + 1)

        self.wA = torch.randn(self.d_in, generator=self._train_gen, dtype=torch.float32)
        self.wB = torch.randn(self.d_in, generator=self._train_gen, dtype=torch.float32)
        self._weights = {"A": self.wA, "B": self.wB}

    def sample(self, task: str, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        if task not in self._weights:
            raise ValueError(f"Unknown task {task!r}. Allowed tasks: A, B")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size!r}")

        x = torch.randn(batch_size, self.d_in, generator=self._train_gen, dtype=torch.float32)
        logits = x @ self._weights[task]
        y = (logits > 0).to(torch.float32).unsqueeze(1)
        return x, y

    def evaluate_accuracy(self, model_fn, task: str, n: int = 512) -> float:
        if task not in self._weights:
            raise ValueError(f"Unknown task {task!r}. Allowed tasks: A, B")
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n!r}")

        x = torch.randn(n, self.d_in, generator=self._eval_gen, dtype=torch.float32)
        logits = x @ self._weights[task]
        y_true = (logits > 0).to(torch.float32).unsqueeze(1)

        with torch.no_grad():
            probs = model_fn(x)
        probs_t = torch.as_tensor(probs, dtype=torch.float32).reshape(n, 1)
        y_pred = (probs_t >= 0.5).to(torch.float32)
        return float((y_pred == y_true).to(torch.float32).mean().item())

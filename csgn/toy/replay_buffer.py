from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity!r}")

        self.capacity = int(capacity)
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._task_ids: torch.Tensor | None = None

        self._task_to_id: dict[str, int] = {}
        self._id_to_task: list[str] = []

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def _ensure_storage(self, d_in: int) -> None:
        if self._x is not None:
            if self._x.shape[1] != d_in:
                raise ValueError(f"Inconsistent x feature size: expected {self._x.shape[1]}, got {d_in}")
            return

        self._x = torch.zeros(self.capacity, d_in, dtype=torch.float32, device="cpu")
        self._y = torch.zeros(self.capacity, 1, dtype=torch.float32, device="cpu")
        self._task_ids = torch.zeros(self.capacity, dtype=torch.long, device="cpu")

    def _task_id(self, task: str) -> int:
        if task not in self._task_to_id:
            self._task_to_id[task] = len(self._id_to_task)
            self._id_to_task.append(task)
        return self._task_to_id[task]

    def add(self, x: torch.Tensor, y: torch.Tensor, task: str) -> None:
        x_cpu = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        y_cpu = torch.as_tensor(y, dtype=torch.float32, device="cpu")

        if x_cpu.ndim == 1:
            x_cpu = x_cpu.unsqueeze(0)
        if y_cpu.ndim == 0:
            y_cpu = y_cpu.reshape(1, 1)
        elif y_cpu.ndim == 1:
            y_cpu = y_cpu.unsqueeze(1)

        if x_cpu.ndim != 2:
            raise ValueError(f"x must be 2D [B, d_in], got shape {tuple(x_cpu.shape)}")
        if y_cpu.ndim != 2 or y_cpu.shape[1] != 1:
            raise ValueError(f"y must be [B, 1], got shape {tuple(y_cpu.shape)}")
        if x_cpu.shape[0] != y_cpu.shape[0]:
            raise ValueError(f"Batch mismatch: x has {x_cpu.shape[0]}, y has {y_cpu.shape[0]}")

        self._ensure_storage(x_cpu.shape[1])
        assert self._x is not None and self._y is not None and self._task_ids is not None

        tid = self._task_id(task)
        for i in range(x_cpu.shape[0]):
            idx = self._ptr
            self._x[idx] = x_cpu[i]
            self._y[idx] = y_cpu[i]
            self._task_ids[idx] = tid

            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size!r}")

        assert self._x is not None and self._y is not None and self._task_ids is not None
        idx = torch.randint(0, self._size, (batch_size,), device="cpu")
        x = self._x[idx].clone()
        y = self._y[idx].clone()
        task_ids = self._task_ids[idx].tolist()
        tasks = [self._id_to_task[int(i)] for i in task_ids]
        return x, y, tasks

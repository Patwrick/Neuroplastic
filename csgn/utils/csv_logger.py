from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        if not fieldnames:
            raise ValueError("fieldnames must be a non-empty list")

        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        if needs_header:
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_row(self, row: dict) -> None:
        payload = {name: row.get(name, "") for name in self.fieldnames}
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(payload)

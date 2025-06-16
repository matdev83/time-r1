from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass(order=True)
class _Entry:
    score: float
    step: int
    path: Path = field(compare=False)


class CheckpointManager:
    """Keeps top-k checkpoints based on a score."""

    def __init__(self, directory: str, k: int = 3) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self._heap: List[_Entry] = []

    def save(self, state: Any, score: float, step: int) -> Path:
        path = self.dir / f"ckpt_{step}.pt"
        with open(path, "wb") as f:
            f.write(b"stub" if isinstance(state, bytes) else b"0")
        entry = _Entry(score, step, path)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        else:
            if score > self._heap[0].score:
                worst = heapq.heapreplace(self._heap, entry)
                worst.path.unlink(missing_ok=True)
            else:
                path.unlink(missing_ok=True)
                return self._heap[0].path
        return path

    @property
    def checkpoints(self) -> List[Path]:
        return sorted(e.path for e in self._heap)

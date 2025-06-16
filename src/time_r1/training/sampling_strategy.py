from __future__ import annotations

import random
from typing import List, Sequence, TypeVar

T = TypeVar("T")


class LocalRandomSampler:
    """Selects samples from a local neighbourhood."""

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def sample(self, items: Sequence[T], k: int) -> List[T]:
        if len(items) <= k:
            return list(items)
        idx = random.randint(0, len(items) - 1)
        choices = []
        for _ in range(k):
            off = random.randint(-self.radius, self.radius)
            choices.append(items[(idx + off) % len(items)])
        return choices


class ClusterRandomSampler:
    """Naive cluster sampler that groups items by index range."""

    def __init__(self, clusters: int = 4) -> None:
        self.clusters = clusters

    def sample(self, items: Sequence[T], k: int) -> List[T]:
        if len(items) <= k:
            return list(items)
        cluster_size = max(1, len(items) // self.clusters)
        selected: List[T] = []
        for i in range(self.clusters):
            start = i * cluster_size
            end = min(len(items), start + cluster_size)
            if start >= len(items):
                break
            bucket = items[start:end]
            if bucket:
                selected.append(random.choice(bucket))
                if len(selected) == k:
                    break
        while len(selected) < k:
            selected.append(random.choice(items))
        return selected

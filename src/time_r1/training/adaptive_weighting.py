from __future__ import annotations

from typing import Iterable

import numpy as np


def adaptive_weights(rewards: Iterable[float]) -> np.ndarray:
    """Softmax over rewards as adaptive weights."""
    arr = np.asarray(list(rewards), dtype=float)
    exp = np.exp(arr)
    weights = exp / exp.sum()
    return weights

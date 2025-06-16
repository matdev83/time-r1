from __future__ import annotations

from typing import Sequence

import numpy as np


def gamma_format(output: str) -> float:
    """Binary reward for proper <think> and <answer> tags."""
    valid = all(
        tag in output for tag in ("<think>", "</think>", "<answer>", "</answer>")
    )
    return 0.0 if valid else -1.0


def gamma_length(answer: str, ground_truth: str) -> float:
    """Scaled reward encouraging complete answers."""
    if len(ground_truth) == 0:
        return 0.0
    ratio = min(len(answer) / len(ground_truth), 1.0)
    return 0.1 * ratio


def gamma_mse(pred: Sequence[float], target: Sequence[float]) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    mse = float(np.mean((pred_arr - target_arr) ** 2))
    return (1 - 1 / (1 + np.exp(-0.3 * mse))) * 2


def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def gamma_seasonal(true: Sequence[float], pred: Sequence[float]) -> float:
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    trend_true = _moving_average(true_arr)
    trend_pred = _moving_average(pred_arr)
    seasonal_true = true_arr - trend_true
    seasonal_pred = pred_arr - trend_pred
    return float(np.mean((seasonal_true - seasonal_pred) ** 2))


def gamma_trend(true: Sequence[float], pred: Sequence[float]) -> float:
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    trend_true = _moving_average(true_arr)
    trend_pred = _moving_average(pred_arr)
    return float(np.mean((trend_true - trend_pred) ** 2))


def _extrema(seq: np.ndarray) -> tuple[list[int], list[int]]:
    max_idx: list[int] = []
    min_idx: list[int] = []
    for i in range(1, len(seq) - 1):
        if seq[i - 1] < seq[i] > seq[i + 1]:
            max_idx.append(i)
        if seq[i - 1] > seq[i] < seq[i + 1]:
            min_idx.append(i)
    return max_idx, min_idx


def gamma_cp(true: Sequence[float], pred: Sequence[float]) -> float:
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    gmax, gmin = _extrema(true_arr)
    cmax, cmin = _extrema(pred_arr)

    def ratio(a: list[int], b: list[int]) -> float:
        return (len([i for i in a if i in b]) / max(1, len(b))) * 0.2

    return ratio(cmax, gmax) + ratio(cmin, gmin)


def total_reward(
    output: str,
    answer: str,
    pred_seq: Sequence[float],
    true_seq: Sequence[float],
) -> float:
    r = gamma_format(output)
    r += gamma_length(answer, answer)
    r += gamma_mse(pred_seq, true_seq)
    r += gamma_seasonal(true_seq, pred_seq)
    r += gamma_trend(true_seq, pred_seq)
    r += gamma_cp(true_seq, pred_seq)
    return float(max(min(r, 1.0), -1.0))

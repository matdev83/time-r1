import numpy as np
import pytest

from time_r1.training.reward import (
    gamma_cp,
    gamma_format,
    gamma_length,
    gamma_mse,
    gamma_seasonal,
    gamma_trend,
    total_reward,
)


def test_format_reward():
    assert gamma_format("<think>x</think><answer>y</answer>") == 0.0
    assert gamma_format("missing") == -1.0


def test_length_reward():
    assert gamma_length("abcd", "abcd") == pytest.approx(0.1)
    assert gamma_length("ab", "abcd") == pytest.approx(0.05)


def test_mse_reward_bounds():
    r = gamma_mse([0.0, 0.0], [0.0, 0.0])
    assert 0.0 <= r <= 1.0


def test_seasonal_trend_cp():
    x = np.arange(5, dtype=float)
    assert gamma_seasonal(x, x) == pytest.approx(0.0)
    assert gamma_trend(x, x) == pytest.approx(0.0)
    assert gamma_cp(x, x) == pytest.approx(0.0)


def test_total_reward_bounds():
    r = total_reward("<think>x</think><answer>y</answer>", "y", [1.0], [1.0])
    assert -1.0 <= r <= 1.0

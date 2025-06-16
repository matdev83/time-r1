import numpy as np

from time_r1.training.adaptive_weighting import adaptive_weights


def test_adaptive_weights_sum_one():
    rewards = [0.1, 0.2, 0.3]
    w = adaptive_weights(rewards)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]

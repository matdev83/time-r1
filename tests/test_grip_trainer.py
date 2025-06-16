import numpy as np
import pandas as pd

from time_r1.training.grip_trainer import GRIPTrainer


def make_df(n: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    data = {
        "open": np.linspace(0, 1, n),
        "high": np.linspace(0.1, 1.1, n),
        "low": np.linspace(-0.1, 0.9, n),
        "close": np.linspace(0.05, 1.05, n),
        "volume": np.linspace(10, 20, n),
    }
    df = pd.DataFrame(data)
    df["timestamp"] = ts
    return df


def test_grip_training_converges():
    df = make_df()
    trainer = GRIPTrainer(
        df, seq_len=5, batch_size=4, lr=0.01, epochs=5, ckpt_dir="tmp_ckpt"
    )
    metrics = trainer.fit()
    assert metrics.kl < 0.1

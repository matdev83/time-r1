import numpy as np
import pandas as pd

from time_r1.training.sft_trainer import SFTTrainer


def make_df(n: int = 120) -> pd.DataFrame:
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


def test_sample_shapes():
    df = make_df()
    trainer = SFTTrainer(df, seq_len=5, batch_size=4, epochs=1)
    batch = next(iter(trainer.dataloader))
    x, y = batch
    assert x.shape[1:] == (5, 5)
    assert y.shape[0] == x.shape[0]


def test_lora_parameter_count():
    df = make_df()
    trainer = SFTTrainer(df, seq_len=5, batch_size=4, epochs=1)
    ratio = trainer.lora_parameter_ratio
    assert ratio < 0.01


def test_training_converges():
    df = make_df()
    trainer = SFTTrainer(df, seq_len=5, batch_size=8, lr=0.02, epochs=8)
    final_loss = trainer.fit()
    assert final_loss < 0.05

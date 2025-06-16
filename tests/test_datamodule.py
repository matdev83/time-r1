import pandas as pd
from pathlib import Path

from time_r1.datasets import build_continuous, parse_contract_filename, NQDataModule

FILES = sorted(Path("data/NQ").glob("NQ *.Last.txt"), key=parse_contract_filename)[:2]


def create_sample_parquet(tmp_path: Path) -> Path:
    df = build_continuous(FILES).head(1000)
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)
    return path


def test_datamodule_shapes(tmp_path):
    parquet_path = create_sample_parquet(tmp_path)
    dm = NQDataModule(str(parquet_path), seq_len=10, batch_size=8)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape[1:] == (10, 5)
    assert batch.shape[0] <= 8


def test_datamodule_lengths(tmp_path):
    parquet_path = create_sample_parquet(tmp_path)
    df = pd.read_parquet(parquet_path)
    dm = NQDataModule(str(parquet_path), seq_len=10, batch_size=8)
    dm.setup()
    total_len = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
    assert total_len == len(df) - 10


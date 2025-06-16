from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_contract_filename(path: Path) -> Tuple[int, int]:
    name = path.stem
    part = name.split()[1]
    month_str, year_str = part.split("-")
    year = 2000 + int(year_str.split(".")[0])
    month = int(month_str)
    return year, month


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    first_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + pd.Timedelta(weeks=2)


def roll_date(year: int, month: int) -> pd.Timestamp:
    return third_friday(year, month) - pd.Timedelta(days=8)


def read_contract(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", names=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S")
    df.sort_values("timestamp", inplace=True)
    return df


def build_continuous(paths: List[Path]) -> pd.DataFrame:
    paths = sorted(paths, key=parse_contract_filename)
    offset = 0.0
    frames: List[pd.DataFrame] = []
    prev_rd = None
    for idx, path in enumerate(paths):
        df = read_contract(path)
        if prev_rd is not None:
            df = df[df["timestamp"] >= prev_rd]
        year, month = parse_contract_filename(path)
        if idx < len(paths) - 1:
            rd = roll_date(year, month)
            prev_close = df[df["timestamp"] < rd]["close"].iloc[-1] + offset
            next_df = read_contract(paths[idx + 1])
            next_df = next_df[next_df["timestamp"] >= rd]
            next_open = next_df.iloc[0]["close"]
            df = df[df["timestamp"] < rd]
            df[["open", "high", "low", "close"]] += offset
            frames.append(df)
            offset += prev_close - next_open
            prev_rd = rd
        else:
            df[["open", "high", "low", "close"]] += offset
            frames.append(df)
    return pd.concat(frames).reset_index(drop=True)


class NQDataset(Dataset):
    """Simple sequence dataset built from the continuous NQ dataframe."""

    def __init__(self, df: pd.DataFrame, seq_len: int) -> None:
        self.seq_len = seq_len
        # keep only numeric columns for tensor conversion
        self.data = (
            df[["open", "high", "low", "close", "volume"]].astype("float32").to_numpy()
        )

    def __len__(self) -> int:  # type: ignore[override]
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        window = self.data[idx : idx + self.seq_len]
        return torch.from_numpy(window)


class NQDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the continuous NQ dataset."""

    def __init__(
        self,
        parquet_file: str,
        seq_len: int = 60,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.parquet_file = parquet_file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_parquet(self.parquet_file)
        n = len(df)
        train_end = int(0.7 * n)
        val_end = int(0.9 * n)
        self.train_ds = NQDataset(
            df.iloc[:train_end].reset_index(drop=True), self.seq_len
        )
        self.val_ds = NQDataset(
            df.iloc[train_end - self.seq_len : val_end].reset_index(drop=True),
            self.seq_len,
        )
        self.test_ds = NQDataset(
            df.iloc[val_end - self.seq_len :].reset_index(drop=True), self.seq_len
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

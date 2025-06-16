from pathlib import Path
from typing import List, Tuple

import pandas as pd

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

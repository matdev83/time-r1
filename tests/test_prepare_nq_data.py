from pathlib import Path

import pytest

from time_r1.datasets.nq import (
    build_continuous,
    parse_contract_filename,
    roll_date,
)

DATA_DIR = Path("data/NQ")
FILES = sorted(DATA_DIR.glob("NQ *.Last.txt"), key=parse_contract_filename)[:2]


@pytest.fixture(scope="module")
def continuous_df(tmp_path_factory):
    df = build_continuous(FILES)
    output = tmp_path_factory.mktemp("out") / "nq.parquet"
    df.to_parquet(output)
    return df


def test_completeness(continuous_df):
    import pandas as pd

    raw_frames = [
        pd.read_csv(f, sep=";", names=["ts", "o", "h", "l", "c", "v"]) for f in FILES
    ]
    for df in raw_frames:
        df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    raw = pd.concat(raw_frames)
    unique_ts = raw.drop_duplicates("ts", keep="last").sort_values("ts")
    assert len(continuous_df) <= len(unique_ts)


def test_no_roll_gaps(continuous_df):
    df = continuous_df
    start_idx = 0
    for file in FILES[:-1]:
        year, month = parse_contract_filename(file)
        rd = roll_date(year, month)
        segment = df[start_idx:]
        prev_close = segment[segment["timestamp"] < rd]["close"].iloc[-1]
        next_close = segment[segment["timestamp"] >= rd]["close"].iloc[0]
        assert abs(prev_close - next_close) < 1e-6
        start_idx = segment[segment["timestamp"] >= rd].index[0]


def test_output_format(continuous_df):
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(continuous_df.columns) == expected_cols
    assert continuous_df["timestamp"].is_monotonic_increasing
    assert not continuous_df.isnull().any().any()

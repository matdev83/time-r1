import pandas as pd
import pytest

from time_r1.datasets import DATASET_SCHEMAS, load_dataset


def test_etth1_loader(tmp_path):
    data = [
        ["2021-01-01 00:00:00", 1, 2, 3, 4, 5, 6, 7],
        ["2021-01-01 01:00:00", 1, 2, 3, 4, 5, 6, 7],
    ]
    cols = ["date"] + DATASET_SCHEMAS["etth1"].features
    df = pd.DataFrame(data, columns=cols)
    csv_path = tmp_path / "etth1.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset("ETTh1", csv_path)
    assert list(loaded.columns) == ["date"] + DATASET_SCHEMAS["etth1"].features
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_schema_validation_missing_column(tmp_path):
    data = [["2021-01-01 00:00:00", 1, 2, 3, 4, 5, 6]]  # one value short
    cols = ["date"] + DATASET_SCHEMAS["etth1"].features[:-1]
    df = pd.DataFrame(data, columns=cols)
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_dataset("etth1", csv_path)

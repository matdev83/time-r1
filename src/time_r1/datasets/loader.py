from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel


class DatasetSchema(BaseModel):
    """Simple schema describing a time series dataset."""

    timestamp: Optional[str]
    features: List[str]

    def check(self, df: pd.DataFrame) -> None:
        cols = [self.timestamp] + self.features if self.timestamp else self.features
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if self.timestamp:
            df[self.timestamp] = pd.to_datetime(df[self.timestamp])
            if df[self.timestamp].isnull().any():
                raise ValueError("Invalid timestamps detected")


DATASET_SCHEMAS: Dict[str, DatasetSchema] = {
    "etth1": DatasetSchema(
        timestamp="date",
        features=[
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT",
        ],
    ),
    "exchange": DatasetSchema(
        timestamp=None,
        features=[f"rate{i}" for i in range(8)],
    ),
}


def load_dataset(name: str, path: str | Path) -> pd.DataFrame:
    """Load a dataset by name and validate against its schema."""

    key = name.lower()
    if key not in DATASET_SCHEMAS:
        raise KeyError(f"Unknown dataset '{name}'")
    schema = DATASET_SCHEMAS[key]
    p = Path(path)
    read_kwargs: Dict[str, object] = {}
    if p.suffix == ".gz":
        read_kwargs["compression"] = "gzip"
        read_kwargs["header"] = None
    elif p.suffix == ".csv":
        if schema.timestamp is None:
            read_kwargs["header"] = None
    df = pd.read_csv(p, **read_kwargs)
    if schema.timestamp is None:
        df.columns = schema.features
        df.insert(0, "timestamp", range(len(df)))
        schema = DatasetSchema(timestamp="timestamp", features=schema.features)
    schema.check(df)
    df = df[[schema.timestamp] + schema.features]
    df.sort_values(schema.timestamp, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

import argparse
from pathlib import Path

from time_r1.datasets.loader import load_dataset

DEFAULT_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert ETTh1 dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="CSV source URL")
    parser.add_argument(
        "--output", default="data/ETTh1.parquet", help="Output parquet file"
    )
    args = parser.parse_args()

    df = load_dataset("ETTh1", args.url)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

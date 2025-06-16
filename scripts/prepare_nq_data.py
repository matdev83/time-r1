import argparse
from pathlib import Path

from time_r1.datasets.nq import build_continuous


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare continuous NQ futures data")
    parser.add_argument("--input-dir", default="data/NQ", help="Directory with raw NQ contract files")
    parser.add_argument("--output", default="data/NQ_continuous.parquet", help="Output parquet file")
    args = parser.parse_args()
    paths = list(Path(args.input_dir).glob("NQ *.Last.txt"))
    if not paths:
        raise FileNotFoundError("No contract files found")
    df = build_continuous(paths)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

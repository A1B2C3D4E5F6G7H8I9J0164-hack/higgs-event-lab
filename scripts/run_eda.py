"""Generate reproducible EDA tables and figures from a HIGGS sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs_event_lab.eda import run_eda


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/higgs_sample_250k.csv.gz")
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--output-dir", default="outputs/eda")
    args = parser.parse_args(argv)
    report = run_eda(args.data, rows=args.rows, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

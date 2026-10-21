"""Streaming smoke test for the raw UCI HIGGS file.

Reads only the first N rows (default 100,000) and checks structure, labels,
missing values, class balance and b-tag values. Standard library only.

Usage:
    python tests/integration/smoke_check_raw_file.py data/raw/HIGGS.csv.gz --rows 100000

Exit code 0 = all checks passed, 1 = at least one check failed.
Prints measured values only; it never prints invented reference numbers.
"""

import argparse
import csv
import gzip
import io
import math
import sys
from collections import Counter
from pathlib import Path

EXPECTED_COLUMNS = 29
N_LOW_LEVEL = 21
N_HIGH_LEVEL = 7
BTAG_COLUMNS = (9, 13, 17, 21)


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def check_file(path, max_rows):
    problems = []
    n_rows = 0
    labels = Counter()
    btag_values = {col: Counter() for col in BTAG_COLUMNS}
    n_bad_width = 0
    n_non_numeric = 0
    n_non_finite = 0
    n_empty = 0
    first_row_looks_like_header = False

    with open_text(path) as handle:
        reader = csv.reader(handle)
        for row in reader:
            if n_rows == 0:
                try:
                    float(row[0])
                except (ValueError, IndexError):
                    first_row_looks_like_header = True
                    continue
            if n_rows >= max_rows:
                break
            n_rows += 1

            if len(row) != EXPECTED_COLUMNS:
                n_bad_width += 1
                continue
            if any(cell.strip() == "" for cell in row):
                n_empty += 1
                continue
            try:
                values = [float(cell) for cell in row]
            except ValueError:
                n_non_numeric += 1
                continue
            if any(not math.isfinite(v) for v in values):
                n_non_finite += 1
                continue

            labels[values[0]] += 1
            for col in BTAG_COLUMNS:
                if len(btag_values[col]) <= 50:
                    btag_values[col][round(values[col], 4)] += 1

    if first_row_looks_like_header:
        problems.append("First row looks like a header; UCI file is expected to have none.")
    if n_rows == 0:
        problems.append("No data rows were read.")
    if n_bad_width:
        problems.append(f"{n_bad_width} rows do not have {EXPECTED_COLUMNS} columns.")
    if n_empty:
        problems.append(f"{n_empty} rows contain empty cells (missing values).")
    if n_non_numeric:
        problems.append(f"{n_non_numeric} rows contain non-numeric values.")
    if n_non_finite:
        problems.append(f"{n_non_finite} rows contain NaN or infinity.")
    unexpected = set(labels) - {0.0, 1.0}
    if unexpected:
        problems.append(f"Unexpected label values: {sorted(unexpected)}.")
    if len(labels) < 2 and n_rows > 0:
        problems.append("Only one class present in the sample; check sampling rule.")

    return {
        "rows_read": n_rows,
        "labels": dict(sorted(labels.items())),
        "btag_distinct_counts": {col: len(c) for col, c in btag_values.items()},
        "btag_values_first_jet": dict(sorted(btag_values[9].items())),
        "problems": problems,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Path to HIGGS.csv or HIGGS.csv.gz")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows to inspect")
    args = parser.parse_args(argv)

    if not Path(args.path).exists():
        print(f"FAIL: file not found: {args.path}")
        return 1

    report = check_file(args.path, args.rows)
    total = sum(report["labels"].values())

    print(f"File               : {args.path}")
    print(f"Rows inspected     : {report['rows_read']}")
    print(f"Label counts       : {report['labels']}")
    if total:
        signal = report["labels"].get(1.0, 0)
        print(f"Signal fraction    : {signal / total:.4f}  (measured on this sample only)")
    print(f"Distinct b-tag values per jet column (capped at 51): {report['btag_distinct_counts']}")
    print(f"Jet 1 b-tag values : {report['btag_values_first_jet']}")

    if report["problems"]:
        print("\nRESULT: FAIL")
        for problem in report["problems"]:
            print(f" - {problem}")
        return 1
    print("\nRESULT: PASS (structure, labels, missing values)")
    print("Reminder: this checks only the first rows. Compare with a later chunk to test sampling bias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

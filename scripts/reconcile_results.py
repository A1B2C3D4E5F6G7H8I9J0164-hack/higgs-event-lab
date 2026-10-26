"""Audit split integrity and saved predictions before finalising the report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs_event_lab.features import ALL_COLUMNS
from higgs_event_lab.protocol import SplitConfig, create_stratified_splits


def _sha256_and_rows(path: Path) -> tuple[str, int]:
    """Hash the complete raw file and count physical CSV rows in one pass."""
    digest = hashlib.sha256()
    rows = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            rows += chunk.count(b"\n")
    return digest.hexdigest(), rows


def _partition_overlap_report(features: pd.DataFrame, split) -> dict[str, int | bool]:
    """Check index disjointness and exact duplicate feature rows between partitions."""
    partitions = {"train": split.train, "validation": split.validation, "test": split.test}
    index_overlap = sum(
        len(set(partitions[left]).intersection(partitions[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    hashes = pd.util.hash_pandas_object(features, index=False).to_numpy()
    sets = {name: set(hashes[indices].tolist()) for name, indices in partitions.items()}
    duplicate_overlap = sum(
        len(sets[left].intersection(sets[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    return {
        "index_overlap_count": int(index_overlap),
        "duplicate_feature_rows_across_partitions": int(duplicate_overlap),
        "partitions_disjoint": index_overlap == 0 and duplicate_overlap == 0,
    }


def _prediction_report(paths: Iterable[Path]) -> dict[str, object]:
    """Recompute accuracy and confirm the Protocol A test index is shared."""
    rows: list[dict[str, object]] = []
    test_index_sets: list[set[int]] = []
    for path in sorted(paths):
        frame = pd.read_csv(path)
        required = {"row_index", "class_label", "score"}
        if not required <= set(frame.columns):
            continue
        test_index_sets.append(set(frame["row_index"].astype(int)))
        rows.append(
            {
                "file": str(path),
                "rows": int(len(frame)),
                "accuracy_from_predictions": float(np.mean((frame["score"] >= 0.5) == frame["class_label"])),
            }
        )
    common_test_indices = len(set.intersection(*test_index_sets)) if test_index_sets else 0
    return {
        "files_checked": len(rows),
        "shared_test_index_count": int(common_test_indices),
        "runs": rows,
    }


def audit(sample_path: Path, prediction_glob: str, raw_path: Path | None = None) -> dict[str, object]:
    """Run T1, T2 and T7 checks and save machine-readable evidence."""
    frame = pd.read_csv(sample_path, header=None, names=list(ALL_COLUMNS))
    labels = frame["class_label"].to_numpy(dtype=np.int64)
    split = create_stratified_splits(labels, SplitConfig(seed=42))
    report: dict[str, object] = {
        "sample": {"path": str(sample_path), "rows": int(len(frame)), "columns": int(frame.shape[1])},
        "split": split.metadata(),
        "partition_audit": _partition_overlap_report(frame.drop(columns=["class_label"]), split),
        "prediction_audit": _prediction_report(Path(path) for path in ROOT.glob(prediction_glob)),
    }
    if raw_path is not None:
        checksum, rows = _sha256_and_rows(raw_path)
        report["raw_file"] = {"path": str(raw_path), "rows": rows, "sha256": checksum, "size_bytes": raw_path.stat().st_size}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/raw/HIGGS.csv.gz")
    parser.add_argument("--raw", default=None, help="Optional raw HIGGS.csv.gz for exact row count and full SHA-256")
    parser.add_argument("--prediction-glob", default="outputs/*_*/test_predictions.csv")
    parser.add_argument("--output", default="outputs/reconciliation_report.json")
    args = parser.parse_args(argv)
    report = audit(Path(args.sample), args.prediction_glob, Path(args.raw) if args.raw else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

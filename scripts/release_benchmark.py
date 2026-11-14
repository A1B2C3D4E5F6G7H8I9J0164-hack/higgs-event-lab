"""Release benchmark: classical baselines against shallow and deep networks.

Reads the raw UCI HIGGS file once, draws nested seeded samples
(250k inside 1M inside 2.5M) and trains, on every budget and feature group:

    logreg       Logistic Regression
    xgboost      XGBoost with early stopping on the validation split
    mlp          the mid-semester compact MLP (64-32), a "shallow" network
    deep_mlp     5 x 300 MLP, the depth and width of Baldi et al. (2014)
    resnet_mlp   residual MLP (Gorishniy et al. 2021)

Every model at a given budget sees the same `higgs-split-v1` train/validation/test
indices. Integrity checks run per budget:

    T1  duplicate events are dropped before sampling, split indices are disjoint,
        and no identical event is in both train and test
    T4  two CPU trainings of the compact MLP with the same seed give the same test scores
    T5  every budget comes from this one script, so comparisons are like for like

Results are written after every run, so a timeout still leaves usable output.

    python scripts/release_benchmark.py --source /path/to/HIGGS.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import xgboost  # noqa: F401  load xgboost's OpenMP runtime before torch's; the other order segfaults on macOS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs_event_lab import classical, neural  # noqa: E402
from higgs_event_lab.features import ALL_FEATURES, FEATURE_GROUPS  # noqa: E402
from higgs_event_lab.protocol import SplitConfig, create_stratified_splits  # noqa: E402

BUDGETS = {"250k": 250_000, "1m": 1_000_000, "2.5m": 2_500_000}
GROUPS = ("low_level", "high_level", "all")
MODELS = ("logreg", "xgboost", "mlp", "deep_mlp", "resnet_mlp")
NET_PRESET = {"mlp": "compact", "deep_mlp": "deep", "resnet_mlp": "resnet"}


def load_raw(path: str) -> np.ndarray:
    import pyarrow.csv as pv

    t0 = time.time()
    table = pv.read_csv(
        path,
        read_options=pv.ReadOptions(autogenerate_column_names=True, block_size=1 << 26),
        convert_options=pv.ConvertOptions(column_types={f"f{i}": "float32" for i in range(29)}),
    )
    if table.num_columns != 29:
        raise SystemExit(f"expected 29 columns, found {table.num_columns}")
    data = np.column_stack([table.column(i).to_numpy() for i in range(29)])
    if not np.isfinite(data).all():
        raise SystemExit("raw file contains non-finite values")
    print(f"loaded {len(data):,} events in {time.time() - t0:.0f}s", flush=True)
    return data


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def split_checks(X: np.ndarray, y: np.ndarray, split) -> dict:
    train, val, test = (set(split.train.tolist()), set(split.validation.tolist()), set(split.test.tolist()))
    raw = np.ascontiguousarray(X)
    hashes = np.array([hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in raw])
    return {
        "rows": int(len(y)),
        "signal_fraction": round(float(y.mean()), 5),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "indices_disjoint": not (train & val or train & test or val & test),
        "covers_every_row": len(train | val | test) == len(y),
        "duplicate_events_in_sample": int(len(hashes) - len(set(hashes.tolist()))),
        "events_shared_by_train_and_test": len(set(hashes[split.train].tolist()) & set(hashes[split.test].tolist())),
        "train_signal_fraction": round(float(y[split.train].mean()), 5),
        "test_signal_fraction": round(float(y[split.test].mean()), 5),
    }


def train_one(model: str, X: np.ndarray, y: np.ndarray, split, seed: int, device: str) -> tuple[dict, object, str]:
    if model == "logreg":
        config = classical.LogisticRegressionConfig(seed=seed)
        return classical.train_logistic_regression(X, y, split=split, config=config), config, "logistic_regression"
    if model == "xgboost":
        config = classical.XGBoostConfig(n_estimators=3000, learning_rate=0.1, max_depth=8,
                                         early_stopping_rounds=50, seed=seed, device=device)
        return classical.train_xgboost(X, y, split=split, config=config), config, "xgboost"
    name = NET_PRESET[model]
    config = neural.preset(name, seed=seed, device=device)
    return neural.train_mlp(X, y, split=split, config=config), config, neural.MODEL_NAMES[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="raw HIGGS.csv (or .csv.gz)")
    ap.add_argument("--out", default="outputs/release")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS), choices=list(BUDGETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "seed": args.seed, "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "budgets": {},
    }

    def save() -> None:
        (out / "release_summary.json").write_text(json.dumps(summary, indent=2))

    data = load_raw(args.source)
    summary["source_rows"] = int(len(data))
    # The raw file repeats some events verbatim. Keep the first copy of each so the
    # same event can never land in both train and test.
    rows_view = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1]))).ravel()
    _, first = np.unique(rows_view, return_index=True)
    unique_rows = np.sort(first)
    summary["duplicate_rows_removed"] = int(len(data) - len(unique_rows))
    print(f"removed {summary['duplicate_rows_removed']:,} duplicate events", flush=True)
    largest = max(BUDGETS[b] for b in args.budgets)
    order = unique_rows[np.random.default_rng(args.seed).permutation(len(unique_rows))[:largest]]
    column = {name: i + 1 for i, name in enumerate(ALL_FEATURES)}

    for tag in sorted(args.budgets, key=BUDGETS.get):
        rows = BUDGETS[tag]
        index = np.sort(order[:rows])
        X_all, y = data[index, 1:], data[index, 0].astype(np.int64)
        split = create_stratified_splits(y, SplitConfig(seed=args.seed))
        budget = summary["budgets"].setdefault(tag, {"rows": rows, "results": []})
        budget["sample_index_sha256"] = hashlib.sha256(index.tobytes()).hexdigest()
        budget["split_checks"] = split_checks(X_all, y, split)
        print(f"[{tag}] {budget['split_checks']}", flush=True)
        save()

        for group in GROUPS:
            cols = [column[name] - 1 for name in FEATURE_GROUPS[group]]
            X = np.ascontiguousarray(X_all[:, cols])
            for model in args.models:
                run_id = f"{model}_{group}"
                t0 = time.time()
                result, config, model_name = train_one(model, X, y, split, args.seed, device)
                run_dir = out / tag / run_id
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)
                common = dict(run_dir=run_dir, run_id=run_id, feature_group=group,
                              feature_names=list(FEATURE_GROUPS[group]),
                              data_source=f"UCI HIGGS seeded sample ({tag}, seed {args.seed})",
                              row_count=rows, config=config, model_name=model_name)
                writer = neural.write_run_artifacts if model in NET_PRESET else classical.write_run_artifacts
                writer(result, **common)
                test = result["test"]
                metrics = test["metrics"]
                row = {
                    "run": run_id, "model": model, "group": group,
                    "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"], "brier": metrics["brier_score"],
                    "ece": expected_calibration_error(np.asarray(test["labels"]), np.asarray(test["probabilities"])),
                    "train_seconds": result.get("training_seconds"),
                    "wall_seconds": round(time.time() - t0, 1),
                    "epochs": len(result.get("history", [])) or None,
                    "parameters": result.get("parameter_count"),
                }
                budget["results"] = [r for r in budget["results"] if r["run"] != run_id] + [row]
                save()
                print(f"[{tag}] {run_id:24s} ROC-AUC {row['roc_auc']:.4f}  "
                      f"PR-AUC {row['pr_auc']:.4f}  Brier {row['brier']:.4f}  {row['wall_seconds']}s", flush=True)

    if "250k" in args.budgets:
        index = np.sort(order[:BUDGETS["250k"]])
        X, y = data[index, 1:], data[index, 0].astype(np.int64)
        config = neural.preset("compact", seed=args.seed, device="cpu", max_epochs=3)
        a = neural.train_mlp(X, y, config=config)["test"]["probabilities"]
        b = neural.train_mlp(X, y, config=config)["test"]["probabilities"]
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        summary["determinism"] = {"device": "cpu", "epochs": 3, "max_abs_probability_diff": diff, "identical": diff == 0.0}

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    save()
    print(json.dumps({k: v for k, v in summary.items() if k != "budgets"}, indent=2))


if __name__ == "__main__":
    main()

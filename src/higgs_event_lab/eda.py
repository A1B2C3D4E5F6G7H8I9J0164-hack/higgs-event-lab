"""Reusable exploratory analysis for the HIGGS benchmark.

The report must cite measurements produced from a real sample, not values typed
into a notebook.  This module computes the EDA tables and saves small JSON/CSV
and PNG artifacts while the input itself remains ignored by git.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from higgs_event_lab.data import load_higgs_sample
from higgs_event_lab.features import FEATURE_GROUPS


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar values into JSON-compatible Python values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _feature_summary(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return distribution, hygiene and class-conditional statistics per feature."""
    signal = X.iloc[y == 1]
    background = X.iloc[y == 0]
    rows: list[dict[str, Any]] = []
    for name in X.columns:
        values = X[name].to_numpy(dtype=np.float64)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        outlier_mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        rows.append(
            {
                "feature": name,
                "count": int(len(values)),
                "unique_count": int(X[name].nunique(dropna=False)),
                "missing_count": int(X[name].isna().sum()),
                "nonfinite_count": int((~np.isfinite(values)).sum()),
                "zero_fraction": float(np.mean(values == 0.0)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(np.max(values)),
                "iqr_outlier_fraction": float(np.mean(outlier_mask)),
                "signal_mean": float(signal[name].mean()),
                "background_mean": float(background[name].mean()),
            }
        )
    return pd.DataFrame(rows)


def _top_correlations(X: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return the strongest absolute Pearson correlations without duplicate pairs."""
    corr = X.corr(method="pearson")
    rows: list[dict[str, Any]] = []
    for left in range(len(corr.columns)):
        for right in range(left + 1, len(corr.columns)):
            value = float(corr.iloc[left, right])
            rows.append(
                {
                    "feature_1": corr.columns[left],
                    "feature_2": corr.columns[right],
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False).head(limit)


def _btag_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact b-tag cardinality, zero mass and most common values."""
    rows: list[dict[str, Any]] = []
    for name in [column for column in X.columns if "btag" in column]:
        counts = X[name].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "feature": name,
                "unique_count": int(counts.size),
                "zero_fraction": float((X[name] == 0.0).mean()),
                "top_values": [
                    {"value": float(value), "count": int(count)}
                    for value, count in counts.head(10).items()
                ],
                "distinct_values": [float(value) for value in sorted(counts.index.tolist())],
            }
        )
    return pd.DataFrame(rows)


def _save_plots(X: pd.DataFrame, y: np.ndarray, feature_summary: pd.DataFrame, output: Path) -> list[str]:
    """Save report-ready EDA figures and return their relative names."""
    signal = X.iloc[y == 1]
    background = X.iloc[y == 0]
    names: list[str] = []

    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Background (0)", "Signal (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#5f6b7a", "#24558f"])
    ax.set_ylabel("Events")
    ax.set_title("Class balance in the EDA sample")
    fig.tight_layout()
    fig.savefig(output / "class_balance.png", dpi=160)
    plt.close(fig)
    names.append("class_balance.png")

    columns = list(X.columns)
    fig, axes = plt.subplots(7, 4, figsize=(14, 20))
    for axis, name in zip(axes.flat, columns):
        axis.hist(background[name], bins=40, density=True, alpha=0.55, label="Background", color="#5f6b7a")
        axis.hist(signal[name], bins=40, density=True, alpha=0.55, label="Signal", color="#24558f")
        axis.set_title(name, fontsize=9)
        axis.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Feature distributions by class", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "feature_distributions.png", dpi=140)
    plt.close(fig)
    names.append("feature_distributions.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(X.corr(method="pearson"), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(columns)), columns, fontsize=6)
    ax.set_title("Pearson correlation matrix")
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    names.append("correlation_heatmap.png")

    grouped = []
    for group, features in FEATURE_GROUPS.items():
        values = feature_summary[feature_summary["feature"].isin(features)]
        pooled = np.abs(values["signal_mean"] - values["background_mean"])
        grouped.append(float(pooled.mean()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Low-level (21)", "High-level (7)", "All (28)"], grouped, color="#24558f")
    ax.set_ylabel("Mean absolute class mean difference")
    ax.set_title("Feature-group exploratory separation")
    fig.tight_layout()
    fig.savefig(output / "feature_group_separation.png", dpi=160)
    plt.close(fig)
    names.append("feature_group_separation.png")
    return names


def save_eda_outputs(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    output_dir: str | Path = "outputs/eda",
    data_source: str = "unknown",
) -> dict[str, Any]:
    """Compute and persist all EDA evidence needed by the report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(y, dtype=np.int64)
    if len(X) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("X and y must have matching rows and both binary labels")

    feature_summary = _feature_summary(X, labels)
    btag_summary = _btag_summary(X)
    correlations = _top_correlations(X)
    feature_summary.to_csv(output / "feature_summary.csv", index=False)
    btag_summary.to_json(output / "btag_summary.json", orient="records", indent=2)
    correlations.to_csv(output / "top_correlations.csv", index=False)
    figures = _save_plots(X, labels, feature_summary, output)

    counts = pd.Series(labels).value_counts().sort_index()
    report: dict[str, Any] = {
        "schema_version": "higgs-eda-v1",
        "data_source": data_source,
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "label_counts": {str(int(key)): int(value) for key, value in counts.items()},
        "signal_fraction": float(np.mean(labels == 1)),
        "missing_values": int(X.isna().sum().sum()),
        "nonfinite_values": int((~np.isfinite(X.to_numpy(dtype=np.float64))).sum()),
        "feature_groups": {name: len(features) for name, features in FEATURE_GROUPS.items()},
        "top_correlations": correlations.to_dict(orient="records"),
        "figures": figures,
        "tables": ["feature_summary.csv", "btag_summary.json", "top_correlations.csv"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")
    return report


def run_eda(
    data_path: str | Path,
    *,
    rows: int = 250_000,
    output_dir: str | Path = "outputs/eda",
) -> dict[str, Any]:
    """Load a documented sample through the validated chunked loader and save EDA."""
    batch = load_higgs_sample(data_path, n_rows=rows, feature_group="all")
    return save_eda_outputs(batch.X, batch.y, output_dir=output_dir, data_source=str(data_path))

# Reusable Python Package (`src/higgs_event_lab/`)

This is the home for reusable, tested package code for the Higgs Event Lab benchmark suite. All modules are importable, unit-tested, and designed to enforce schema correctness and leakage-free splitting.

## Module Reference

| Module | Status | Owner | Responsibilities |
|---|---|---|---|
| `features.py` | Implemented (MID-01) | Aditya Rana | Verified UCI schema, feature group mappings (`low_level`, `high_level`, `all`), label exclusion safeguards, `get_feature_columns()`. |
| `data.py` | Implemented (MID-01) | Aditya Rana | Memory-safe chunked loader (`iter_higgs_chunks`), schema/type/value validation (`validate_raw_chunk`), reproducible sampling (`extract_sample`), interim staging (`stage_interim_clean_data`). |
| `protocol.py` | Implemented (MID-05) | Krishna | Shared deterministic 3-way holdout protocol (`higgs-split-v1`), `higgs-experiment-v1` artifact contract (`new_experiment_artifact`). |
| `splits.py` | Implemented (MID-05) | Krishna | Explicit reproducible split policy and indices wrapper (`SplitConfig`, `SplitIndices`, `create_stratified_splits`). |
| `artifacts.py` | Implemented (MID-05) | Krishna | Versioned artifact schema and export helpers shared with the dashboard (`write_run_artifacts`). |
| `evaluation.py` | Implemented (MID-05) | Krishna | Standardized ROC-AUC, PR-AUC, Brier score, calibration curves, validation threshold selection (`compute_classification_metrics`). |
| `neural.py` | Implemented (MID-04) | Rachit Gupta | Compact PyTorch MLP with BatchNorm, Dropout, early stopping, multi-group benchmark runner, artifact writing (`build_mlp`, `train_mlp`, `write_run_artifacts`). |
| `classical.py` | Implemented (MID-03) | Aditya Rana | Logistic Regression baseline wrapper with StandardScaler pipeline, threshold tuning, multi-group benchmarks. |

## Key Design Principles

- **Label safety:** The target label (`class_label`, column 0) is never placed inside the feature matrix `X`.
- **Leakage prevention:** Scalers and models are fit on the training split only; validation and test are transformed but never fitted.
- **Deterministic splits:** All splits use seed 42 under `higgs-split-v1` (70% train / 15% validation / 15% test).
- **Strict validation:** `validate_raw_chunk` rejects non-numeric, NaN, infinite, or invalid-label data immediately — no silent row dropping.
- **Schema contract:** Experiment artifacts conform to `higgs-experiment-v1` and are validated by the dashboard loader.

## Ownership

- **Aditya Rana** owns `data.py`, `features.py`, `classical.py`.
- **Rachit Gupta** owns `neural.py`.
- **Krishna** owns `protocol.py`, `splits.py`, `artifacts.py`, `evaluation.py`, and deployment.

Keep module boundaries small and keep tests close to the behavior they validate.

## Usage Example

```python
from pathlib import Path
from higgs_event_lab.data import load_higgs_sample
from higgs_event_lab.protocol import create_stratified_splits, SplitConfig
from higgs_event_lab.neural import MLPConfig, train_mlp, write_run_artifacts

# Load a validated sample
batch = load_higgs_sample("data/processed/higgs_sample_250k.csv.gz", n_rows=50000, feature_group="all")

# Create a leakage-free stratified split
split = create_stratified_splits(batch.y, SplitConfig(seed=42))

# Train a compact MLP
config = MLPConfig(hidden_sizes=(256, 128, 64), dropout=0.30, max_epochs=60, seed=42)
result = train_mlp(batch.X, batch.y, split=split, config=config)

# Export dashboard-compliant artifacts
meta_path, pred_path = write_run_artifacts(
    result,
    run_dir="outputs/wide_mlp_all",
    run_id="wide_mlp_all_seed42",
    feature_group="all",
    feature_names=list(batch.X.columns),
    data_source="data/processed/higgs_sample_250k.csv.gz",
    row_count=len(batch.X),
    config=config,
)
```

---

# Local Model Artifacts

The `artifacts/` directory holds model checkpoints and serialized preprocessing objects locally. These files are **git-ignored** and never committed to source control unless the team later agrees on a documented small-model release policy.

## Purpose

Store trained model artifacts produced by benchmark runs so they can be reloaded for inference, report generation, or the Streamlit Model Simulation view without retraining.

## What Belongs Here

| File Type | Example | Notes |
|---|---|---|
| Model checkpoints | `*.pt`, `*.pth`, `*.ckpt` | PyTorch model state dicts or full models |
| Serialized scikit-learn models | `*.joblib`, `*.pkl` | Logistic Regression, XGBoost models |
| Preprocessing objects | `*.joblib`, `*.pkl` | Fitted `StandardScaler`, encoders |
| NumPy arrays | `*.npy`, `*.npz` | Embeddings, feature matrices |
| Universal Binary JSON | `*.ubj` | Alternative serialization |

## Naming Convention

Use descriptive, versioned names that encode model type, feature group, and seed:

```
<model>_<feature_group>_seed<seed>.<ext>
```

Examples:
- `logistic_regression_all_seed42.joblib`
- `xgboost_high_level_seed42.joblib`
- `wide_mlp_all_seed42.pt`
- `feature_scaler_all_seed42.joblib`

## Retention & Cleanup

- Keep only the artifacts needed for the current report or dashboard simulation.
- Remove obsolete checkpoints when they are superseded by new runs.
- Do not store large prediction tables or full output CSVs here; those belong under `outputs/` (also git-ignored).

## Policy

> **Do not commit model weights to source control unless the team agrees on a documented small-model release policy.** Large binary artifacts bloat the repository and violate GitHub file-size limits. Record model type, data/split metadata, feature group, seed, dependency versions, and metric file in `outputs/full_colab_results.json` or the relevant run's `metrics.json`.

---

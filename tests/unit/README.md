# Unit Tests

Add small, fast tests for one function or invariant at a time. Avoid fetching the full UCI data in unit tests; use synthetic fixtures instead.

## What to Test

| Area | Examples |
|---|---|
| Data schema | 29-column enforcement, label values in `{0, 1}`, no NaN/Inf, rejection of malformed chunks. |
| Label leakage | Target column never appears in `X`; scaler fit on train only. |
| Feature dimensions | `low_level` = 21, `high_level` = 7, `all` = 28; deterministic column order. |
| Splitting | Deterministic stratified splits with seed 42; no overlap between train/val/test. |
| Metrics | ROC-AUC, PR-AUC, Brier score edge cases; calibration curve bin alignment. |
| Artifacts | `higgs-experiment-v1` JSON schema validation; prediction CSV column and bounds checks. |
| Notebook style | Valid nbformat, Markdown before code, explanation length, no tutorial comments, cell length limits. |
| Download | SHA-256 computation, SSL context, skip-download when file exists. |

## Conventions

- One test function per behavior/invariant.
- Use `tmp_path` fixtures for temporary files.
- Keep tests under a few seconds each.
- Name files `test_<module>_<behavior>.py`.

## Files

| File | Purpose |
|---|---|
| `test_data_loader.py` | Chunked reading, schema validation, sampling, manifest creation. |
| `test_download.py` | SHA-256, SSL context, skip-existing-download behavior. |
| `test_protocol.py` | Split configuration, stratification, artifact schema version. |
| `test_features.py` | Feature group mappings, label exclusion, column counts. |
| `test_classical.py` | Logistic Regression pipeline, threshold tuning. |
| `test_neural.py` | MLP architecture, training output shape, artifact export. |
| `test_evaluation.py` | Metric computation, threshold selection, calibration. |
| `test_app_loader.py` | Dashboard artifact discovery and validation. |
| `test_app_metrics.py` | Dashboard metric calculations. |
| `test_app_views.py` | Dashboard view rendering. |
| `test_dashboard_loader.py` | Dashboard loader integration. |
| `test_feature_dictionary.py` | Feature dictionary CSV consistency. |
| `test_notebook_style.py` | Notebook structure and style rules. |
| `test_repo_hygiene.py` | Forbidden files, large files, secrets, broken links, metric-number claims. |

## Running

```bash
# Unit tests only
pytest tests/unit/

# A specific test file
pytest tests/unit/test_data_loader.py -v
```

---

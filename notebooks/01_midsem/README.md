# Mid-Semester MVP Notebooks

This folder contains the mid-semester notebooks. They now read the complete approximately 11-million-row raw HIGGS file directly; the smaller processed samples remain available for optional quick checks. Notebook explanations must be Markdown text cells, not tutorial comments inside code cells.

## Current Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda_and_feature_groups.ipynb` | Exploratory data analysis: schema validation, class balance, missing-value checks, discrete feature verification, kinematic distributions, invariant-mass distributions, correlation structure across feature groups. |
| `02_neural_network_baseline.ipynb` | Compact MLP architecture (BatchNorm, Dropout), feature-group comparison (`low_level`, `high_level`, `all`), reproducible stratified splitting, artifact export for the dashboard. |
| `03_classical_baseline.ipynb` | Full-data Logistic Regression and XGBoost baselines across all three feature groups, with ROC curves and artifact export. |

## Pipeline Summary

1. Load the complete raw file using `higgs_event_lab.data.load_higgs_sample` (chunked, validated loading).
2. Verify schema: 29 columns, binary labels in `{0, 1}`, no NaN or infinite values.
3. Check class balance (signal fraction) and compare against a later chunk to guard against sampling bias.
4. Inspect low-level vs. high-level feature distributions for signal vs. background.
5. Train a compact MLP across feature groups with early stopping and validation-selected thresholds.
6. Export run artifacts (`metrics.json`, `test_predictions.csv`) compliant with `higgs-experiment-v1`.

## Style & Standards

- Use Markdown cells as text boxes; each section states the question, why it matters, what the code does, expected output, and interpretation.
- Avoid tutorial-style comments inside code cells.
- Keep code cells short (max 40 lines); move reusable logic into `src/higgs_event_lab/`.
- Record source, sample count, seed, split, package versions, model settings, and runtime.
- Do not commit notebook outputs containing large data or misleading placeholder scores.

Style is enforced by `tests/unit/test_notebook_style.py`.

---

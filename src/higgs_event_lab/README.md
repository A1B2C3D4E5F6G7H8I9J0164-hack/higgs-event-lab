# Reusable Python package

This is the planned home for reusable, tested code. The scaffold does not implement models yet.

Planned modules, to be added as issues are approved:

- `data.py`: source access, schema checks and sampling/chunking.
- `features.py`: label separation and low-level/high-level/all feature groups.
- `splits.py`: explicit reproducible split policy.
- `classical.py`: Logistic Regression and XGBoost baseline wrappers.
- `neural.py`: compact PyTorch MLP and training.
- `evaluation.py`: metrics, calibration and validation-selected operating point.
- `artifacts.py`: versioned artifact schema shared with the dashboard.
- `config.py`: explicit run configuration.

Aditya owns data/features/classical modules; Rachit owns neural.py; Krishna owns shared split/evaluation/artifact interfaces. Keep module boundaries small and tests close to the behavior they validate.

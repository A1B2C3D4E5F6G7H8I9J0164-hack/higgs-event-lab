# Reusable Python package

This is the planned home for reusable, tested code. The scaffold does not implement models yet.

Modules status:
- `features.py`: **[implemented, MID-01]** verified UCI schema, feature group mappings (`low_level`, `high_level`, `all`), label exclusion safeguards.
- `data.py`: **[implemented, MID-01]** memory-safe chunked loader (`iter_higgs_chunks`), schema/type/value validation (`validate_raw_chunk`), reproducible sampling with later-chunk check (`extract_sample`).
- `splits.py`: explicit reproducible split policy.
- `classical.py`: Logistic Regression and XGBoost baseline wrappers.
- `neural.py`: compact PyTorch MLP and training.
- `evaluation.py`: metrics, calibration and validation-selected operating point.
- `artifacts.py`: versioned artifact schema shared with the dashboard.
- `config.py`: explicit run configuration.

Aditya owns data/features/classical modules; Rachit owns neural.py; Krishna owns shared split/evaluation/artifact interfaces. Keep module boundaries small and tests close to the behavior they validate.

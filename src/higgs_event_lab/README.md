# Reusable Python package

This is the planned home for reusable, tested code. The scaffold does not implement models yet.

Modules status:
- `features.py`: **[implemented, MID-01]** verified UCI schema, feature group mappings (`low_level`, `high_level`, `all`), label exclusion safeguards. *(Aditya)*
- `data.py`: **[implemented, MID-01]** memory-safe chunked loader (`iter_higgs_chunks`), schema/type/value validation (`validate_raw_chunk`), reproducible sampling with later-chunk check (`extract_sample`). *(Aditya)*
- `protocol.py`: **[implemented, MID-05]** shared deterministic 3-way holdout protocol and `higgs-experiment-v1` artifact contract. *(Krishna)*
- `splits.py`: **[implemented, MID-05]** explicit reproducible split policy and indices wrapper. *(Krishna)*
- `artifacts.py`: **[implemented, MID-05]** versioned artifact schema and export helpers shared with the dashboard. *(Krishna)*
- `evaluation.py`: **[implemented, MID-05]** standardized ROC-AUC, PR-AUC, Brier score, calibration curves, and validation threshold selection. *(Krishna)*
- `neural.py`: **[implemented, MID-04]** compact PyTorch MLP with batch norm, dropout, early stopping, and multi-group benchmark runner. *(Rachit)*
- `classical.py`: **[pending: Aditya, MID-03]** Logistic Regression and XGBoost baseline wrappers.

Aditya owns data/features/classical modules; Rachit owns neural.py; Krishna owns shared split/evaluation/artifact interfaces and deployment. Keep module boundaries small and tests close to the behavior they validate.

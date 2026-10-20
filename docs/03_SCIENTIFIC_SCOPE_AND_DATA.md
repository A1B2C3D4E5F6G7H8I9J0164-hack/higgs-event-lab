# Scientific Scope, Dataset and Data Governance

## Physics context

The Higgs boson is an excitation associated with the Higgs field. The project is not a field simulation. It addresses a supervised classification benchmark built from simulated collision events and a label indicating signal/background.

## Dataset boundary

Use the official [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs). The course brief describes 11 million examples, one target label and 28 inputs split into 21 low-level features and 7 expert-derived high-level features. It recommends a subset of 250,000–1,000,000 events to start.

## Sample plan

- **Mid-semester:** target 250,000 rows, or a smaller documented number if compute or connectivity forces it. Record the actual data source and row count.
- **End-semester:** target up to 1,000,000 rows if performance and runtime allow.
- **Stretch:** use the source's documented held-out evaluation protocol if feasible and if the input access method preserves it. Document any deviation; don't describe a subset as the full benchmark.

## Feature groups

The 28 inputs consist of 21 low-level kinematic measurements and 7 expert-derived features according to the course brief. Aditya must verify exact column names/order against UCI documentation and preserve that mapping in a feature dictionary before any model uses the groups. Do not guess column names or assume a split without verifying the official schema.

## Leakage controls

- Keep the target label out of feature matrices and scalers.
- Split the data before fitting preprocessing transforms.
- Fit imputers/scalers only on the training partition.
- Select models, hyperparameters and thresholds using training/validation data only.
- Evaluate final choices once on held-out test data.
- Use the same held-out examples for every model comparison.
- Fix and report random seeds; record package versions and split strategy.

## Required metrics

At minimum: ROC-AUC, PR-AUC/average precision, calibration/Brier score where implemented, and training time. Also report sample size, class balance, test threshold metrics and inference time if feasible. Accuracy alone is insufficient for the project question.

## Product-oriented extension

A validation-selected threshold for target signal efficiency can make background rejection an understandable operating trade-off. Report test-set results at this frozen threshold. Do not calculate or claim particle-discovery significance without the additional physics inputs and formal statistical procedure required for it.

## Data rules

- Download source data locally; do not put it in repository history.
- Never silently fall back to a third-party mirror. If the official source is inaccessible, document the alternative and get Krishna's approval before using it in a report.
- Never publish personal tokens, environment files, or course-private material.
- Store data-source URL, retrieval date, actual row count, target mapping, seed and split metadata with each run.

# Project Charter

## Working title

**Higgs Event Lab: Reproducible Signal-versus-Background Classification and Model Trade-off Workbench**

## Problem statement

The course project asks us to classify simulated particle-collision events as Higgs-associated signal or background. A model score alone is not enough for a defensible project: we need a fair comparison across feature groups and model families, transparent metrics, reproducible data handling, and an explanation of which model is useful under an explicitly stated evaluation objective.

## Primary research question

How do low-level kinematic features, seven expert-derived high-level features, and their combination affect the performance and compute cost of classical ML models versus a compact neural network on a documented HIGGS sample?

## Product question

Can we provide an easy-to-use research workbench that loads a completed experiment's saved results, compares model/feature choices, explains trade-offs and exports a reproducible snapshot?

## Intended users

Course assessors, students learning ML for high-energy physics, and researchers doing exploratory benchmark comparisons. The prototype is educational/research software, not a detector control system or an approved physics analysis tool.

## Required scope from the course brief

- UCI HIGGS simulated event-classification dataset.
- Compare 21 low-level inputs, 7 high-level expert-derived inputs and all 28 inputs.
- Logistic Regression and XGBoost classical baselines.
- A compact MLP with batch normalization and dropout.
- Same held-out evaluation data for fair comparisons; fit preprocessing on training data only.
- Report ROC-AUC, PR-AUC / average precision, calibration and training time.
- Begin with a documented 250,000–1,000,000 event subset; do not start with all 11 million events.

## Planned extension, not a required claim

Choose an operating threshold on validation data for a declared target signal efficiency; report the corresponding background rejection on test data. Add uncertainty estimates for the main model comparison only when runtime permits. This supports an interpretable trade-off but does not equal discovery significance.

## Non-goals

- Proving that the Higgs boson exists; it has already been observed experimentally.
- Simulating the Higgs field or all particle masses.
- Claiming real-data discovery, production detector performance, real-world calibrated probabilities or statistical significance from the benchmark alone.
- Training on the entire dataset before smaller-sample correctness is established.
- Building a general-purpose model-training platform before the required benchmark works.

## Success criteria

1. Data provenance, schema, row count, feature groups, split and seed are recorded.
2. Baselines and MLP are evaluated on the same untouched held-out set.
3. Preprocessing and threshold selection do not use test data.
4. Results are saved as machine-readable artifacts and used consistently by the report and dashboard.
5. A new teammate can reproduce the starter run from the documentation.
6. The dashboard never invents results when no valid experiment artifacts exist.
7. Mid-semester and end-semester claims are scoped to what has actually been run.

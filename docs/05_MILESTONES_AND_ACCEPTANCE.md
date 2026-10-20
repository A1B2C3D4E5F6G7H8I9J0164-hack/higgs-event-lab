# Milestones and Acceptance Criteria

Calendar dates are intentionally blank until Krishna confirms the actual mid-semester and end-semester deadlines.

## Stage 0 — Repository foundation

**Owner:** Krishna. **Support:** all team members; Saumya records decisions and updates docs.

Done when: new repository created, scaffold committed, four collaborators invited, branch/PR rules agreed, initial issue board populated, no data/secrets tracked, every member can make a small PR.

## Stage 1 — Data understanding and scope confirmation

**Lead:** Aditya for schema/data; Saumya for sources and understandable explanations; Krishna signs off.

Done when: source URL cited, target mapping verified, row count and feature schema checked, 21+7 feature mapping documented from the official source, sample plan and split protocol agreed, and a data smoke-test checklist exists.

## Stage 2 — Mid-semester MVP

**Goal:** deliver a modest, reproducible benchmark and explainable demo rather than an unfinished grand system.

- Aditya: EDA, data checks and Logistic Regression/XGBoost baselines on feature groups.
- Rachit: compact MLP prototype and training metadata.
- Krishna: split/metric protocol, comparison contract, integration and presentation.
- Saumya: background/method documentation, QA, issue handling, change log and reproducibility review.
- Ronit: dashboard shell showing measured saved outputs and an honest empty state before outputs exist.

Acceptance: documented source/row count, low/high/all feature experiments as supported by runtime, fair held-out evaluation, ROC-AUC and average precision, calibration/training time where implemented, figures, known limitations and a reproducible demo. Missing experiments must be stated explicitly.

## Stage 3 — Research hardening

**Goal:** verify that the observed differences are trustworthy.

- Repeat key results with chosen seeds.
- Verify group columns and prevent label leakage.
- Add tests for data schema, splits, metrics and artifact schema.
- Add validation-selected threshold analysis for a declared target signal efficiency.
- Record calibration and compute trade-offs; quantify uncertainty for the primary comparison if runtime permits.
- Ensure the final test set was not used for model/threshold selection.

## Stage 4 — Product completion and end-semester submission

- The dashboard reads actual experiment artifacts and shows source, sample count, split and model metadata.
- User can compare model/features, inspect ROC/PR/calibration, and view the validation-selected operating point evaluated on test.
- Exports include metric table and experiment metadata; missing data shows a clear instruction, not fake values.
- Saumya reviews documentation, known issues, change log and setup instructions.
- Krishna reproduces the release run, creates/tag a version and deploys the reviewed build.
- Team produces the report, demo script, limitations section, references and viva preparation.

## Release gate

A release must not proceed if it has fabricated or mismatched metrics, known data leakage, an undocumented data source, broken artifact loading, secrets or raw dataset files, or wording that claims a real discovery.

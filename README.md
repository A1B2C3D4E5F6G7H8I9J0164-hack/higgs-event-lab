# Higgs Event Lab

**A team research project for comparing machine-learning methods on the simulated UCI HIGGS event-classification dataset.**

> **Scientific boundary:** this project classifies simulated examples labelled signal/background. It does not simulate the Higgs field, establish a new discovery, provide a detector-grade analysis, or turn classifier scores into real-world Higgs probabilities.

## Team

- **Krishna — Team Lead, ML Experiment Design, Integration and Deployment**
- **Saumya Mishra — Documentation Lead, QA/Bug Triage and Change Log**
- **Aditya Rana — Dataset Pipeline, EDA and Classical ML Baselines**
- **Rachit Gupta — Deep Learning / MLP Pipeline and Reproducibility**
- **Ronit Singh — Product UI / Streamlit Dashboard and Usability**

The detailed responsibility boundaries and acceptance criteria live in [`team/TEAM_CHARTER.md`](team/TEAM_CHARTER.md).

## Read in this order

1. [`docs/00_START_HERE.md`](docs/00_START_HERE.md)
2. [`docs/01_PROJECT_CHARTER.md`](docs/01_PROJECT_CHARTER.md)
3. [`docs/02_REPOSITORY_STRUCTURE.md`](docs/02_REPOSITORY_STRUCTURE.md)
4. [`team/TEAM_CHARTER.md`](team/TEAM_CHARTER.md)
5. [`docs/03_SCIENTIFIC_SCOPE_AND_DATA.md`](docs/03_SCIENTIFIC_SCOPE_AND_DATA.md)
6. [`docs/04_GITHUB_WORKFLOW.md`](docs/04_GITHUB_WORKFLOW.md)
7. [`docs/05_MILESTONES_AND_ACCEPTANCE.md`](docs/05_MILESTONES_AND_ACCEPTANCE.md)
8. [`docs/06_AI_PROMPTS.md`](docs/06_AI_PROMPTS.md)

## What the assignment requires

The course brief asks for a comparison of signal/background classification using low-level inputs, expert-derived inputs, and all inputs; classical baselines including Logistic Regression and XGBoost; a compact neural network with batch normalization and dropout; held-out evaluation; and reporting ROC-AUC, PR-AUC, calibration and training time. The proposed starter sample is 250,000–1,000,000 events before attempting the full dataset. See the course brief and the linked official source before changing this scope.

## Product direction

The target product is an understandable workbench that lets a user inspect documented experiments and compare model trade-offs. The model should answer a declared task, not simply show a leaderboard: for example, compare background rejection when a target signal efficiency is selected on validation data. This threshold-based operating-point analysis is an extension and must not be confused with discovery significance.

## Important project rules

- Keep raw data and large/generated outputs out of Git.
- Do not commit secrets, API keys, personal tokens, or local environment files.
- Use Markdown cells to explain notebook sections. Avoid tutorial-style comments inside notebook code cells; keep code cells focused.
- The data/model module owner fixes defects in their module. Saumya owns bug intake, reproduction, triage, documentation updates and verification coordination; she is not expected to rewrite every other person's module alone.
- Krishna owns final integration and deployment. No one deploys an unreviewed main branch.
- Never report unrun results or invent metrics. Record actual source, event count, seed, split, configuration and runtime.

## Data source

Use the [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs) and cite the dataset and original paper in any academic report. The course brief describes 21 low-level and 7 expert-derived features (28 inputs total) and recommends beginning with a documented subset. Store downloaded data under `data/raw/` locally, not in the public repository.

## Current status

This repository scaffold defines project scope, folder contracts, team ownership, working conventions and milestones. It deliberately does **not** yet contain completed model-training code or real benchmark scores. Those will be added in reviewed issues/PRs in the next phase.

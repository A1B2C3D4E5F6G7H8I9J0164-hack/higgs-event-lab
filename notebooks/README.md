# Notebooks

This folder contains Jupyter notebooks for the Higgs Event Lab project. Notebooks are organized by milestone: physics introduction, mid-semester MVP, and full-data Colab/Kaggle benchmark.

## Notebook Run Order

Run the notebooks in this order on a fresh machine:

| # | Notebook | Purpose | Data Used |
|---|---|---|---|
| 1 | `00_physics/00_higgs_physics_and_dataset_intro.ipynb` | Beginner Higgs physics and dataset context; no model results | None (read-only dictionary inspection) |
| 2 | `01_midsem/01_eda_and_feature_groups.ipynb` | Exploratory data analysis: schema, class balance, feature distributions, correlations | 50,000 events from processed sample |
| 3 | `01_midsem/02_neural_network_baseline.ipynb` | Compact MLP architecture, feature-group comparison, artifact export | Synthetic / sample data |
| 4 | `03_full/00_full_higgs_colab.ipynb` | Full 1M benchmark: downloads UCI data, trains Logistic Regression, XGBoost, Wide MLP; exports metrics, predictions, and model weights | 1,000,000 events from UCI (downloaded automatically) |

Each notebook downloads the original UCI HIGGS dataset automatically when `data/raw/HIGGS.csv.gz` is missing. The download is large (about 2.6 GB), so the first run takes time. The notebooks also build `data/processed/higgs_sample_250k.csv.gz` or `higgs_sample_1m.csv.gz` when missing.

## Data Sources

- Official UCI archive: <https://archive.ics.uci.edu/dataset/280/higgs>
- Download zip: <https://archive.ics.uci.edu/static/public/280/higgs.zip>

## Notebook Structure Standards

Use Markdown cells as text boxes. Each major section should state:

1. **What question this section answers.**
2. **Why the step is needed.**
3. **What the code will do.**
4. **What output to expect.**
5. **How to interpret it and what could go wrong.**

Keep analytical code cells readable and avoid tutorial-style comments in the code. Do not duplicate utility logic that belongs in `src/higgs_event_lab/`. Record source, sample count, seed, split, package versions, model settings, and runtime. Do not commit notebook outputs that contain large data or misleading placeholder scores.

## Style Enforcement

Notebook style is enforced by `tests/unit/test_notebook_style.py`, which checks:
- Valid nbformat JSON (version >= 4).
- First cell is a Markdown title/overview.
- Every code cell is preceded by a Markdown explanation of at least 25 words.
- Code cells contain no tutorial-style comments.
- Code cells are short enough to read (max 40 lines); move logic into `src/higgs_event_lab/`.
- No large or stale outputs are committed.

---

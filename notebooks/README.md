# Notebooks

Jupyter notebooks for the HiggsEvent project, organised by milestone.

## Run Order (Start Here)

Run notebooks **in this exact order** on a fresh machine. Each one auto-downloads and auto-builds its data requirements.

| # | Notebook | Purpose | Data | Where | Time |
|---|---|---|---|---|---|
| 0 | `00_physics/00_higgs_physics_and_dataset_intro.ipynb` | Physics intro — read only, no model results | None | Local | 1 min |
| 1 | `01_midsem/01_eda_and_feature_groups.ipynb` | EDA: schema, class balance, distributions, correlations on all rows | `data/raw/HIGGS.csv.gz` | Local | 15–30 min (first run) |
| 2 | `01_midsem/02_neural_network_baseline.ipynb` | Compact MLP (PyTorch): BatchNorm, Dropout, artifact export on all rows | `data/raw/HIGGS.csv.gz` | Local/GPU recommended | Depends on hardware |
| 3 | `01_midsem/03_classical_baseline.ipynb` | Logistic Regression + XGBoost on all feature groups and all rows | `data/raw/HIGGS.csv.gz` | Local/GPU recommended | Depends on hardware |
| 4 | `03_full/00_full_higgs_colab.ipynb` | Full 11M-event benchmark: downloads data, trains all models, exports weights and metrics | `data/raw/HIGGS.csv.gz` | **Colab / Kaggle (GPU)** | Long-running |

> **Note:** Notebooks 1–3 can be run locally in sequence without any GPU. Notebook 4 requires Colab or Kaggle for GPU access and handles the large 1M-event training run.

---

## Data Download

The UCI HIGGS dataset (~2.6 GB compressed, approximately 11 million rows) is **automatically downloaded in full** when you run any model notebook. These notebooks use `data/raw/HIGGS.csv.gz` directly for EDA and training. The 250k and 1M files remain available only as optional smaller derived samples.

If you prefer to download it ahead of time:

```bash
# Using the project script
python scripts/download_data.py --output data/raw/HIGGS.csv.gz

# Or directly with curl
curl -L https://archive.ics.uci.edu/static/public/280/higgs.zip -o /tmp/higgs.zip
unzip /tmp/higgs.zip "*.csv.gz" -d data/raw/
```

- **Source:** [UCI ML Repository #280](https://archive.ics.uci.edu/dataset/280/higgs)
- **Complete archive:** <https://archive.ics.uci.edu/static/public/280/higgs.zip>
- **Target path:** `data/raw/HIGGS.csv.gz`
- **Size:** ~2.6 GB compressed, 11 million rows

---

## Folder Structure

```
notebooks/
├── 00_physics/
│   ├── README.md
│   └── 00_higgs_physics_and_dataset_intro.ipynb   ← Read this first
├── 01_midsem/
│   ├── README.md
│   ├── 01_eda_and_feature_groups.ipynb             ← Run 1st (data + EDA)
│   ├── 02_neural_network_baseline.ipynb            ← Run 2nd (MLP)
│   └── 03_classical_baseline.ipynb                 ← Run 3rd (LogReg + XGBoost)
└── 03_full/
    └── 00_full_higgs_colab.ipynb                   ← Run on Colab/Kaggle
```

---

## What Each Notebook Produces

| Notebook | Outputs |
|---|---|
| `01_eda_and_feature_groups.ipynb` | Full-data EDA tables and plots under `outputs/eda/` |
| `02_neural_network_baseline.ipynb` | Full-data `outputs/mlp_*/metrics.json`, predictions and learning curves |
| `03_classical_baseline.ipynb` | Full-data `outputs/logreg_*/`, `outputs/xgboost_*/`, ROC plot |
| `03_full/00_full_higgs_colab.ipynb` | Full-data `outputs/full_colab_results.json`, model weights and test predictions |

After running notebooks 1–3, start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

---

## Style Standards

Notebook style is enforced by `tests/unit/test_notebook_style.py`. Rules:

- First cell must be a Markdown title/overview.
- Every code cell must be preceded by a Markdown explanation (≥ 25 words).
- No tutorial-style inline comments in code cells.
- Code cells must be ≤ 40 lines; move reusable logic into `src/higgs_event_lab/`.
- No large stale outputs committed to Git.

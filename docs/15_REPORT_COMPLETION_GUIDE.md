# Final report completion guide

The attached PDF is a rendered report, so its yellow text cannot be updated
from this repository. Use this guide after running the notebooks and copy only
the measured values from the generated artifacts into the report source.

## Recommended run order

1. **Colab or Kaggle:** open `notebooks/03_full/00_full_higgs_colab.ipynb`.
   Enable internet, run the setup cell, download the official UCI archive, and
   create the one-million-row sample. Keep the raw archive and model weights in
   the runtime or cloud storage; do not commit them.
2. **EDA:** in the repository root run:

   ```bash
   python scripts/run_eda.py --data data/processed/higgs_sample_250k.csv.gz \
       --rows 250000 --output-dir outputs/eda
   ```

   If using Colab, run the equivalent notebook cell with the absolute
   `/content/...` path. Download the `outputs/eda/` folder when the runtime is
   finished.
3. **Reconciliation:** run the commands in `docs/14_RESULTS_RECONCILIATION.md`.
   This produces the exact raw row count, full checksum, split-overlap audit,
   common test-index check and recomputed accuracies.
4. **Full benchmark:** run the corrected training cell in the full notebook.
   The output summary now includes separate Logistic Regression, XGBoost and
   MLP times, operating points and genuine MLP learning history.
5. **Dashboard:** start `streamlit run app/main.py`, capture the four required
   views, and verify the CSV export. Save screenshots outside git or in the
   submission package if the university requires them.
6. **Final checks:** run `python -m pytest tests/unit -q`, then record the real
   output and environment versions in the report appendix.

## Which notebooks to run

| Notebook | Run it? | Report use |
|---|---|---|
| `notebooks/03_full/00_full_higgs_colab.ipynb` | **Yes — required** | Downloads data, creates the 1M/250k samples, generates EDA evidence, audits the split and trains the 1M benchmark. |
| `notebooks/01_midsem/01_eda_and_feature_groups.ipynb` | Optional | Use only if you want an interactive EDA walkthrough. The full Colab notebook now creates the saved EDA files itself. |
| `notebooks/01_midsem/02_neural_network_baseline.ipynb` | Optional / recommended for Protocol A | Use when you need fresh 250k MLP feature-group results or visible epoch history for the 250k report section. |
| `notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb` | Optional | Background explanation for the viva/report; it does not produce model metrics. |

The full Colab notebook includes fallbacks for `run_eda.py` and
`reconcile_results.py`. Therefore it still produces the EDA and reconciliation
artifacts when the cloned GitHub repository does not yet contain those helper
scripts. You only need to run the other notebooks if you want their specific
interactive outputs.

## Yellow-item replacement map

| Report location | Replace the yellow text with |
|---|---|
| Section 8, retrieval date | The date on which the raw file was downloaded in the runtime, in `YYYY-MM-DD` format. |
| Section 8, exact row count | `raw_file.rows` from `outputs/reconciliation_report.json`. |
| Section 8, smoke test | The actual `RESULT: PASS` output and inspected-row count from the smoke script. |
| Section 9, EDA outputs | Tables and figures from `outputs/eda/`; cite `summary.json` as the measurement source. The fallback produces `feature_summary.csv`, `btag_summary.json`, `correlation_matrix.csv`, `class_conditional_means.csv`, `class_balance.csv`, `class_balance.png`, and `correlation_heatmap.png`. |
| Sections 13–14, Protocol B settings | The `training` object in the corrected full benchmark JSON. |
| Section 14, epochs and device | `epochs_completed`, `history` length and `device` from the MLP run. |
| Section 14, learning curves | Plot `training.history` from the full benchmark JSON; never draw a synthetic curve. |
| Section 16, screenshots | Four screenshots from the running Streamlit app. |
| Section 19, PyTorch and raw-file tests | The actual commands and outputs from the final runtime. |
| Section 20, Protocol B times | `training_seconds` from the corrected full benchmark JSON. |
| Section 20, Protocol B operating point | `operating_point` from each full benchmark record. |
| Section 20.4, headline | Krishna's written decision after T1–T7 and the Protocol B-on-250k comparison. |
| Section 17, contributions | One short confirmation paragraph from each team member. |

## Colab commands

Do not start by running the project commands in a blank Colab cell. First run
this one setup cell; it clones and imports the repository, installs the project
requirements, changes into the repository, and sets `HIGGS_REPO_ROOT`:

```python
from pathlib import Path
import os, subprocess, sys

REPO_URL = "https://github.com/CosmicMagnetar/higgs-event-lab.git"
root = Path("/content/higgs-event-lab")
if not (root / "src" / "higgs_event_lab").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True)
os.environ["HIGGS_REPO_ROOT"] = str(root)
os.chdir(root)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, str(root / "src"))
print("Repository imported:", root)
```

Then run these cells:

```python
!python tests/integration/smoke_check_raw_file.py data/raw/HIGGS.csv.gz --rows 100000
!python scripts/run_eda.py --data data/processed/higgs_sample_250k.csv.gz --rows 250000 --output-dir outputs/eda
!python scripts/reconcile_results.py --sample data/processed/higgs_sample_250k.csv.gz --raw data/raw/HIGGS.csv.gz --output outputs/reconciliation_report.json
!python scripts/train_full_colab.py --sample data/processed/higgs_sample_1m.csv.gz --output-dir outputs
```

On Kaggle, set **Internet** to on, use `/kaggle/working/higgs-event-lab` as
`HIGGS_REPO_ROOT`, and download the JSON/CSV/PNG outputs from the notebook
Files panel before the session expires.

## What must be shown in the final PDF

Include the measured EDA summary, at least the class-balance plot, one
distribution figure, the correlation heatmap, the feature-group comparison,
the corrected Protocol B table, MLP learning curves, dashboard screenshots,
the final test table, and a short limitations paragraph. Keep the statement
that HIGGS is simulated and that classifier scores are not a particle-discovery
claim.

# Notebook policy

## Notebook run order

Run the notebooks in this order on a fresh machine.

1. `00_physics/00_higgs_physics_and_dataset_intro.ipynb`
   Beginner physics and dataset context. Contains no model results.
2. `01_midsem/01_eda_and_feature_groups.ipynb`
   Exploratory data analysis on a 50,000 event sample.
3. `01_midsem/02_neural_network_baseline.ipynb`
   Compact MLP architecture and feature-group comparison.

Each notebook downloads the original UCI HIGGS dataset automatically when
`data/raw/HIGGS.csv.gz` is missing. The download is large (about 2.6 GB), so
the first run takes time. The notebooks also build
`data/processed/higgs_sample_250k.csv.gz` when it is missing.

## Data sources

- Official UCI archive: https://archive.ics.uci.edu/dataset/280/higgs
- Download zip: https://archive.ics.uci.edu/static/public/280/higgs.zip

## Notebook structure

Use Markdown cells as text boxes. Each major section should state:

1. What question this section answers.
2. Why the step is needed.
3. What the code will do.
4. What output to expect.
5. How to interpret it and what could go wrong.

Keep analytical code cells readable and avoid tutorial-style comments in the code. Do not duplicate utility logic that belongs in `src/higgs_event_lab/`. Record source, sample count, seed, split, package versions, model settings and runtime. Do not commit notebook outputs that contain large data or misleading placeholder scores.

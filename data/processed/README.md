# Processed Data

Prepared dataset subsets are cached here locally and are not committed to Git to keep the repository lightweight and within GitHub limits.

## External Download Links

For fast local reproduction without extracting from the 2.6 GB raw file, the 250,000-event benchmark sample (`higgs_sample_250k.csv.gz`, ~52 MB) is available:

* **Team Cloud Mirror (Google Drive / Hugging Face):**
  - Google Drive mirror: `[Add your Google Drive link here]`
  - Hugging Face Datasets mirror: `[Add your Hugging Face repo link here]`

## Placement Instructions

Place the processed sample into this folder:
```bash
# Target path:
data/processed/higgs_sample_250k.csv.gz
```

Alternatively, extract it locally from raw data using the project pipeline:
```bash
PYTHONPATH=src python -m higgs_event_lab.data \
  --source data/raw/HIGGS.csv.gz \
  --output data/processed/higgs_sample_250k.csv.gz \
  --rows 250000
```
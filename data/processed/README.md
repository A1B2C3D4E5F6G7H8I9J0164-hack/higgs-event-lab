# Processed Data

Prepared dataset subsets are cached here locally. Manifest files are tracked by Git; the data files themselves are **git-ignored** to keep the repository lightweight and within GitHub limits.

## Available Samples

| Sample | Rows | Approx. Size | File | Status |
|---|---|---|---|---|
| Mid-semester subset | 250,000 | ~52 MB compressed | `higgs_sample_250k.csv.gz` | Generated via pipeline |
| Full benchmark subset | 1,000,000 | ~200 MB compressed | `higgs_sample_1m.csv.gz` | Generated via Colab notebook |

## External Download Links

For fast local reproduction without extracting from the 2.6 GB raw file:

- **Team Cloud Mirror (Google Drive / Hugging Face):**
  - Google Drive mirror: `[Add your Google Drive link here]`
  - Hugging Face Datasets mirror: `[Add your Hugging Face repo link here]`

## Placement Instructions

Place the processed sample into this folder:

```bash
# Target path for 250k sample:
data/processed/higgs_sample_250k.csv.gz

# Target path for 1M sample:
data/processed/higgs_sample_1m.csv.gz
```

## Generate Locally from Raw Data

Alternatively, extract samples locally from raw data using the project pipeline:

```bash
# 250,000-event mid-semester sample
PYTHONPATH=src python -m higgs_event_lab.data \
  --source data/raw/HIGGS.csv.gz \
  --output data/processed/higgs_sample_250k.csv.gz \
  --rows 250000

# 1,000,000-event full benchmark sample (requires raw data downloaded first)
PYTHONPATH=src python -m higgs_event_lab.data \
  --source data/raw/HIGGS.csv.gz \
  --output data/processed/higgs_sample_1m.csv.gz \
  --rows 1000000
```

Each extraction writes a sidecar manifest file (e.g., `higgs_sample_1m.csv.gz.manifest.json`) recording source path, row counts, label distribution, signal fraction, sampling method, random seed, and creation timestamp.

## Policy

- Processed data files are **not committed to Git** (tracked only via `.gitignore` exceptions for READMEs and committed result manifests).
- Do not manually edit processed CSVs; regenerate them from raw source using the documented pipeline to preserve reproducibility.
- Record the source URL, retrieval date, archive/file name, checksum, dataset schema, and citation when introducing a new sample.

---

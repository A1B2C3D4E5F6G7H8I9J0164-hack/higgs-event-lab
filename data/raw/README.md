
# Raw Data

Download the official UCI HIGGS dataset locally into this folder. The raw data are **not committed to Git** due to GitHub's 100 MB file size limit and repository hygiene rules.

## External Download Links

* **Official UCI Repository:** [UCI Machine Learning Repository: HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) (`HIGGS.csv.gz`, 2.6 GB)
* **Team Cloud Mirror (Google Drive / Hugging Face):**
  - Google Drive mirror: `[Add your Google Drive link here]`
  - Hugging Face Datasets mirror: `[Add your Hugging Face repo link here]`

## Placement Instructions

Place the downloaded file into this directory without uncompressing or renaming:
```bash
# Target path:
data/raw/HIGGS.csv.gz
```

Verify integrity with SHA-256 checksum:
```bash
shasum -a 256 data/raw/HIGGS.csv.gz
```

## Requirements & Guidelines

* **Documentation:** Record the source URL, retrieval date, archive/file name, checksum, dataset schema, and citation in a data card.
* **Source Integrity:** Do not silently replace the source with an unverified mirror.

---

## References

* See `docs/09_DATASET_GUIDE.md` for the verified schema.
* See `docs/11_SETUP_GUIDE.md` for download and smoke-test steps.


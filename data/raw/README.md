# Raw Data

Download the official UCI HIGGS dataset locally into this folder. The raw data are **not committed to Git** due to GitHub's 100 MB file size limit and repository hygiene rules. Only this README is tracked.

---

## External Download Links

| Source | URL | File | Size |
|---|---|---|---|
| **Official UCI Repository** | <https://archive.ics.uci.edu/dataset/280/higgs> | `HIGGS.csv.gz` | ~2.6 GB |
| **Direct Download Zip** | <https://archive.ics.uci.edu/static/public/280/higgs.zip> | `higgs.zip` | ~2.6 GB |
| **Team Cloud Mirror** | `[Add your Google Drive link here]` | — | — |
| **Team Cloud Mirror** | `[Add your Hugging Face repo link here]` | — | — |

---

## Placement Instructions

Place the downloaded file into this directory **without uncompressing or renaming**:

```bash
# Target path:
data/raw/HIGGS.csv.gz
```

---

## Verify Integrity

After download, verify the file with a SHA-256 checksum:

```bash
shasum -a 256 data/raw/HIGGS.csv.gz
```

Record the checksum, source URL, retrieval date, and archive name in a data card for provenance.

## Requirements & Guidelines

| Guideline | Requirement |
|---|---|
| **Documentation** | Record source URL, retrieval date, archive/file name, checksum, dataset schema, and citation in a data card. |
| **Source Integrity** | Do not silently replace the source with an unverified mirror. |
| **Format** | Keep the file gzip-compressed (`HIGGS.csv.gz`) and headerless (29 columns per row, no header row). |
| **Size** | The raw archive contains approximately 11 million rows (simulated collision events). |

## Dataset Citation

> Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications*, 5, 4308. [DOI: 10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308).

## References

- See `docs/09_DATASET_GUIDE.md` for the verified schema.
- See `docs/11_SETUP_GUIDE.md` for download and smoke-test steps.

---

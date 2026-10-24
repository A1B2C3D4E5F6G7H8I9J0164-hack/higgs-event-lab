# Interim Data

The `data/interim/` directory is reserved for intermediate, staged, or temporary data transformations during pipeline execution.

> **Policy:** All data files in this directory are local and **never committed to Git**. The `.gitignore` configuration explicitly excludes `data/interim/*` while tracking this documentation.

---

## 1. Role in the Data Pipeline

Interim data serves three specific roles in the project workflow:
1. **Chunk Staging & Verification:** Staging streaming chunks during full dataset validation sweeps or sampling before compiling into final subsets.
2. **Intermediate Transformations:** Holding temporary scaled features, outlier filters, or alternative representation experiments before committing to `data/processed/`.
3. **Partition Caching:** Staging index lists or temporary partition dumps during hyperparameter sweeps to avoid recomputing preprocessing on every iteration.

---

## 2. Operating Guidelines

* **Full Reproducibility:** Every intermediate file generated in `data/interim/` must be deterministically reproducible from raw source data using a script or command with documented parameters and random seeds.
* **Row Count & Dropped Record Accounting:** If any rows are filtered or modified, the generating code must log the exact before/after row count and the reason for rejection (e.g., non-finite entries or corrupted sensors).
* **No Test Set Fitting:** Preprocessing fits (such as scalers or encoders) must never be applied to interim data before splitting into train/validation/test partitions.
* **Periodic Cleanup:** Intermediate files should be removed once final processed subsets are exported to `data/processed/` to save local disk space.

---

## 3. Recommended Code Pattern

When writing intermediate chunk outputs, use explicit chunk naming:

```python
from pathlib import Path
import pandas as pd

interim_dir = Path("data/interim")
interim_dir.mkdir(parents=True, exist_ok=True)

# Example: Staging an inspected chunk
chunk_path = interim_dir / "chunk_0000_validated.csv"
df_chunk.to_csv(chunk_path, index=False)
```

To purge temporary files:
```bash
# Clean up temporary interim cache
rm -f data/interim/*.csv data/interim/*.parquet data/interim/*.tmp
```
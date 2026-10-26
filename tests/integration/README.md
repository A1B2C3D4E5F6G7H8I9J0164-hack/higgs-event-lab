# Integration Tests

Integration tests confirm that modules exchange the documented fields and output schema end to end, using tiny deterministic synthetic datasets.

## What Is Tested

- End-to-end flow: data loading → stratified splitting → model training → metrics → artifact writing.
- Dashboard artifact loading and validation (`higgs-experiment-v1` contract).
- Smoke checks for small real-source fetches when network access is available.

## Policy

- Use tiny deterministic synthetic datasets so tests run quickly and offline.
- Never present synthetic test output as experiment evidence.
- Keep tests focused on module boundaries and schema compatibility, not on model performance claims.

## Files

| File | Purpose |
|---|---|
| `test_data_pipeline.py` | Sample extraction, reproducibility, manifest creation from synthetic fixtures. |
| `test_end_to_end_pipeline.py` | Full flow: load → split → train MLP → export artifacts → validate schema. |
| `smoke_check_raw_file.py` | Optional smoke test for real-source data access. |

---

# Tests

Tests are added alongside implementation, not postponed until the deadline. The suite enforces repository hygiene, leak prevention, schema integrity, and artifact validity.

## Test Structure

| Folder | Purpose |
|---|---|
| `unit/` | Small, fast tests for one function or invariant at a time: data schema checks, label/feature separation, feature-group shape, metric calculations, threshold choice, artifact validation, notebook style, download logic. |
| `integration/` | Deterministic synthetic-data flows through loading, preparation, model prediction, metrics, output writing, and dashboard loading. |

## Synthetic Data Policy

Synthetic data are for testing code paths only. They must **not** be described as HIGGS benchmark results or presented as experiment evidence.

## Running Tests

```bash
# Run the full suite (104 tests)
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v
```

## Coordination

Saumya Mishra coordinates the QA checklist. Module owners write and repair tests for the code they own:
- Aditya Rana: data, features, classical tests.
- Rachit Gupta: neural tests.
- Krishna: protocol, splits, evaluation, artifacts, dashboard, repo hygiene tests.

---

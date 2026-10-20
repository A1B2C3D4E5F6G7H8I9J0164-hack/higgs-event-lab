# Tests

Tests are added alongside implementation, not postponed until the deadline.

- `unit/`: data schema checks, label/feature separation, feature-group shape, metric calculations, threshold choice and artifact validation.
- `integration/`: small deterministic synthetic-data flow through loading/preparation, model prediction, metrics, output writing and dashboard loading.

Synthetic data are for testing code paths only. They must not be described as HIGGS benchmark results. Saumya coordinates the QA checklist; module owners write/repair tests for the code they own.

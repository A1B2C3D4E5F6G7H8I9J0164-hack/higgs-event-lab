# Team Charter — Five People, Clear Ownership

## Principles

Each member owns a deliverable that can be demonstrated, not just a vague support role. Each module has one accountable owner, one reviewer and a documented interface. The team leader coordinates and integrates rather than doing everyone's implementation. Documentation/QA is a real workstream and bug intake is managed centrally, but module owners remain accountable for fixing defects in their code.

## Ownership table

### Krishna — Team Lead, Research Protocol, Integration and Deployment

**Owns:** overall research question and scope; repository decisions; experiment/split/metric/artifact contracts; cross-module integration; final report synthesis; release checklist; deployment and rollback plan.

**Must deliver:** approved project charter, locked data/split protocol, artifact schema accepted by Aditya/Rachit/Ronit, integration branches, release notes approval and deployed application.

**Does not own:** all classical/MLP implementation or all documentation edits. He must review them and resolve integration decisions.

### Saumya Mishra — Documentation Lead, QA/Bug Triage and Change Documentation

**Owns:** readable documentation, source/reference hygiene, beginner-friendly explanations, notebook Markdown text boxes, issue reproduction/triage, known-issues list, change log and documentation release checklist. Coordinates regression validation and directly fixes documentation, test/checklist, packaging and low-risk cross-cutting bugs.

**Must deliver:** source and terminology guide; updated docs for every user-visible change; reproducibility checklist; bug queue with reproduction and owner; change log; documentation sign-off before release.

**Bug ownership rule:** Saumya leads the process and verifies fixes, but Aditya fixes root causes in data/classical code, Rachit in MLP code, Ronit in app code, and Krishna in integration/deployment code. This distributes code expertise and prevents Saumya becoming a bottleneck or being blamed for code she does not own.

### Aditya Rana — Dataset Pipeline, EDA and Classical ML

**Owns:** official source/schema checks, feature dictionary for low-level/high-level/all groups, data validation, chunked sample loading, EDA, Logistic Regression, XGBoost baseline, baseline metrics and timing.

**Must deliver:** data card, reproducible loader/preparation, EDA plots, classical results/artifacts, data tests and run metadata. Must explain any sample/protocol deviation.

### Rachit Gupta — Deep Learning / MLP and Reproducibility

**Owns:** compact PyTorch MLP with batch normalization/dropout, training loop, validation/early stopping settings, device/runtime tracking, model prediction artifacts and neural-network diagnostics.

**Must deliver:** trained-run metadata, reproducible training configuration, validation/test predictions conforming to artifact schema, model tests and a clear comparison to classical models.

### Ronit Singh — Product UI / Streamlit Dashboard

**Owns:** navigation/layout, results loading, model comparison charts, threshold/operating-point view, user errors/empty states, artifact validation in the UI, and app usability tests.

**Must deliver:** working dashboard driven by real saved results; visible provenance and limitations; export/view workflow; setup instructions and app smoke test. Does not invent metrics or train models on app startup.

## Shared obligations

- Everyone writes or maintains tests for code they own.
- Everyone records actual commands/results when closing issues.
- Every member reviews at least one teammate PR before the end-semester release.
- Every substantive change includes documentation updates; Saumya checks consistency.
- All team members must be able to explain the core research question, feature groups, metrics, leakage prevention and limitations at the viva.

## Cross-review map

- Krishna reviews the experiment protocol and integration boundaries; does a release review of everyone's work.
- Saumya checks scientific wording, usability docs, regression checklist and change notes; she is not the only code reviewer.
- Aditya reviews Rachit's feature/split compatibility and metric parity.
- Rachit reviews Aditya's model-comparison contract and output consistency.
- Ronit reviews the readability/usability of documentation with Saumya and provides feedback on the artifact schema early, not after model code is finished.

## Decision rule

If members disagree, write the two options, impact, evidence and recommendation in an issue or ADR. Krishna decides project integration and release direction; scientific claims must be checked against the source and may not be overridden by convenience.

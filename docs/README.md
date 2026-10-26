# Documentation Index

This folder contains the full project documentation for **HiggsEvent**. Documents are numbered for reading order.

| # | Document | Description | Audience |
|---|---|---|---|
| [00](00_START_HERE.md) | START HERE | Where to begin: reading order, quick-start commands, team onboarding | Everyone |
| [01](01_PROJECT_CHARTER.md) | Project Charter | Goals, scope, scientific boundaries, success criteria | Everyone |
| [02](02_REPOSITORY_STRUCTURE.md) | Repository Structure | Folder layout, file naming conventions, what is tracked vs. git-ignored | Developers |
| [03](03_SCIENTIFIC_SCOPE_AND_DATA.md) | Scientific Scope & Data | Research questions, dataset overview, what the project does and does not claim | Everyone |
| [04](04_GITHUB_WORKFLOW.md) | GitHub Workflow | Branch naming, PR process, commit message format, code review checklist | Developers |
| [05](05_MILESTONES_AND_ACCEPTANCE.md) | Milestones & Acceptance | Mid-semester and end-semester deliverables, acceptance criteria | Everyone |
| [06](06_AI_PROMPTS.md) | AI Prompts | Approved prompts for using AI assistants, boundary rules, audit log | Developers |
| [07](07_REFERENCES.md) | References | Full bibliography with DOIs, dataset citations, software citations | Everyone |
| [08](08_PHYSICS_BACKGROUND.md) | Physics Background | Higgs boson, LHC, decay signatures, kinematic observables, invariant masses | Everyone |
| [09](09_DATASET_GUIDE.md) | Dataset Guide | UCI HIGGS schema, 29-column layout, feature dictionary, class balance | Data/ML |
| [10](10_OBJECTIVES_AND_METHODOLOGY.md) | Objectives & Methodology | Feature group definitions, model architectures, evaluation protocol, threshold selection | ML |
| [11](11_SETUP_GUIDE.md) | Setup Guide | Step-by-step local install, data download, smoke tests, environment setup | Developers |
| [12](12_QA_TEST_PLAN.md) | QA & Test Plan | 105-test suite overview, CI policy, data integrity checks, notebook style rules | QA/Dev |
| [13](13_EXPERIMENT_PROTOCOL_AND_ARTIFACTS.md) | Experiment Protocol | `higgs-experiment-v1` artifact schema, metrics.json structure, predictions CSV | ML/Dev |
| [14](14_RESULTS_RECONCILIATION.md) | Results Reconciliation | Recorded benchmark numbers, reconciliation notes, known discrepancies | Everyone |
| [15](15_REPORT_COMPLETION_GUIDE.md) | Report Completion Guide | Template for the final mid-semester report, section checklist, graph requirements | Everyone |
| — | [feature_dictionary.csv](feature_dictionary.csv) | Machine-readable feature metadata (name, type, group, PDG source, description) | ML/Data |
| — | [decisions/](decisions/) | Architecture Decision Records (ADRs) | Developers |

---

## Architecture Decision Records (ADRs)

| ADR | Status | Decision |
|---|---|---|
| [ADR-0001](decisions/ADR-0001-data-source-and-project-boundary.md) | Accepted | Use UCI HIGGS (280) as sole data source; no real detector data |
| [DRAFT-ADR-0002](decisions/DRAFT-ADR-0002-experiment-artifact-schema.md) | Draft | `higgs-experiment-v1` schema contract for all benchmark runs |

---

## Quick Links for Mid-Semester

- **Physics background:** [08_PHYSICS_BACKGROUND.md](08_PHYSICS_BACKGROUND.md)
- **How to get the data:** [11_SETUP_GUIDE.md](11_SETUP_GUIDE.md)
- **Feature definitions:** [09_DATASET_GUIDE.md](09_DATASET_GUIDE.md)
- **What models to run:** [10_OBJECTIVES_AND_METHODOLOGY.md](10_OBJECTIVES_AND_METHODOLOGY.md)
- **Benchmark results:** [14_RESULTS_RECONCILIATION.md](14_RESULTS_RECONCILIATION.md)
- **Report template:** [15_REPORT_COMPLETION_GUIDE.md](15_REPORT_COMPLETION_GUIDE.md)

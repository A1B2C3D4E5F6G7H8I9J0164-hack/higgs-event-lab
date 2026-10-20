# ADR-0001: Official Dataset Source and Project Boundary

- **Status:** Accepted for project setup
- **Decision owner:** Krishna, with team review

## Decision

The baseline source is the official UCI HIGGS dataset. The work is simulated-event binary classification and a research/product workbench, not a Higgs-field simulator or particle-discovery claim. Begin with the course-brief-recommended subset scale and increase sample size only after the pipeline is correct and runtime is measured.

## Consequences

- Aditya verifies the schema/feature mapping from UCI documentation.
- All data source changes must be recorded and approved; no silent mirrors.
- Mid-semester and end-semester results can have different sample sizes, but each must report its actual count.
- Source data and large run artifacts remain outside Git.
- Threshold-selection analysis is an extension; thresholds are chosen on validation, then evaluated on the test split.

## Revisit when

The official source cannot be accessed, the instructor requires a different split/protocol, or the actual assignment rubric imposes additional requirements. Record any revision as a new ADR instead of silently changing this file.

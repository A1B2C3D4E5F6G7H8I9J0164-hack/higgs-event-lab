# `01_PROJECT_CHARTER.md`

## Project Charter

### Working Title

**Higgs Event Lab:** *Reproducible Signal-versus-Background Classification and Model Trade-off Workbench*

---

### Problem Statement

The course project requires classifying simulated particle-collision events as signal or background. In the HIGGS benchmark, the signal represents a hypothetical exotic-Higgs process and the background represents a top-quark process with matching final-state particles (Baldi et al., 2014; see `docs/08_PHYSICS_BACKGROUND.md`, Section 5).

> **Note:** This project is **not** a re-creation of the 2012 ATLAS/CMS Higgs boson discovery.

A standalone model score is insufficient for a defensible analysis. This project focuses on establishing a fair comparison across feature groups and model families, transparent evaluation metrics, reproducible data pipelines, and a rationale for model selection based on explicitly declared evaluation objectives.

---

### Core Questions

* **Primary Research Question:**
How do low-level kinematic features, seven expert-derived high-level features, and their combination affect performance and compute cost across classical ML models versus a compact neural network on a documented HIGGS sample?
* **Product Question:**
Can we provide an easy-to-use research workbench that loads saved experiment artifacts, compares model/feature trade-offs, and exports reproducible execution snapshots?

---

### Target Audience

Course assessors, students learning ML applications in high-energy physics (HEP), and researchers performing exploratory benchmark comparisons.

> **Disclaimer:** This prototype is educational/research software — it is not a detector control system or an approved physics analysis tool.

---

### Scope Breakdown

#### Required Scope (Course Brief)

* **Dataset:** UCI HIGGS simulated event-classification dataset.
* **Feature Sets:** Compare performance across:
1. 21 low-level kinematic inputs
2. 7 high-level expert-derived inputs
3. All 28 inputs combined


* **Models:**
* **Classical Baselines:** Logistic Regression, XGBoost
* **Deep Learning:** Compact Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout


* **Evaluation Protocol:** Use a fixed held-out evaluation dataset; fit all preprocessing transforms exclusively on training data.
* **Metrics:** Report ROC-AUC, PR-AUC / Average Precision, model calibration, and training time.
* **Dataset Scaling:** Begin with a documented **250,000–1,000,000 event subset** (do not process the full 11-million-row dataset initially).

#### Planned Extension *(Optional / Secondary)*

Select an operating decision threshold on validation data for a declared target signal efficiency, then report the corresponding background rejection on test data. Add uncertainty estimates for primary model comparisons if runtime budget allows. *(Supports interpretable trade-off analysis without claiming statistical discovery significance.)*

---

### Non-Goals

* ❌ **Experimental Validation:** Proving the existence of the Higgs boson (already experimentally verified).
* ❌ **Full Simulation:** Simulating the full Higgs field or fundamental particle mass generation.
* ❌ **Real-World Claims:** Claiming real-data discovery, production detector performance, or statistically calibrated physical probabilities.
* ❌ **Premature Scale:** Training on all 11 million events before small-sample correctness is verified.
* ❌ **Over-engineering:** Building a generic model-training framework before the baseline HIGGS benchmark is fully operational.

---

### Success Criteria

* [ ] Data provenance, schema, row count, feature groupings, split definitions, and random seeds are fully recorded.
* [ ] All baselines and neural networks are evaluated on the identical, untouched held-out test split.
* [ ] Data preprocessing and operating threshold selections strictly exclude test data.
* [ ] Metrics and outputs are persisted as machine-readable artifacts and used consistently by reports and dashboards.
* [ ] A new team member can successfully reproduce the baseline pipeline using the project documentation.
* [ ] The interactive dashboard displays only verified experiment artifacts and never fabricates results.
* [ ] Project milestones (mid-semester / end-semester) report exclusively on completed execution runs.

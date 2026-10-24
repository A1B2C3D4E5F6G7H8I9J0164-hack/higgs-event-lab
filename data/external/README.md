# External Data

The `data/external/` directory is reserved for auxiliary reference data, alternative benchmarks, or external cross-checks.

> **Governance Policy (ADR-0001):** The official UCI HIGGS dataset is the primary ground truth. No external dataset or third-party mirror may be used or mixed into benchmark evaluations without explicit justification and team approval. Files in this folder are local and **never committed to Git** (`data/external/*` is git-ignored).

---

## 1. Permitted Uses of External Data

1. **Schema and Physical Cross-Checks:** Comparing derived feature formulas (e.g. invariant mass calculations) against open CERN ROOT libraries or particle physics reference tables.
2. **Alternative Standard Model Benchmarks:** Cross-evaluating model generalization on related collision simulation benchmarks (such as SUSY or HEPMASS) for research extensions.
3. **Verified Mirrors:** If the primary UCI server is unreachable, a verified, bit-for-bit identical mirror may be downloaded into this directory after SHA-256 verification against the UCI ground truth.

---

## 2. Intake and Provenance Template

Any external dataset placed in this folder must be recorded in this README using the following format:

### External Source Record [EXT-001]: Official UCI HIGGS Benchmark Reference

* **Dataset / Asset Name:** UCI Machine Learning Repository HIGGS Benchmark (Dataset ID 280)
* **Source URL:** <https://archive.ics.uci.edu/dataset/280/higgs>
* **Retrieval Date:** 2026-10-21
* **License & Attribution:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  - *Citation:* Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications*, 5, 4308. [DOI: 10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308).
* **SHA-256 Checksum:** Documented in project data manifest.
* **Schema Comparison:**
  - Column count: 29 columns (1 label + 21 low-level kinematic variables + 7 high-level invariant masses).
  - Schema alignment: Exact 1-to-1 match with [`docs/feature_dictionary.csv`](../../docs/feature_dictionary.csv).
  - Discrepancies vs primary: None.
* **Rationale for Inclusion:** Defines the benchmark ground truth for feature indexing, units, coordinate systems (transverse momentum, pseudorapidity, azimuth), and signal/background definition.
* **Lead Owner:** Aditya Rana (Dataset & Classical ML Lead)
* **Approval Sign-off:** Krishna (Team Lead) — *Status: Verified & Approved*


---

## 3. Strict Prohibitions

* **No Silent Substitution:** Never silently substitute the primary `data/raw/HIGGS.csv.gz` with an external file.
* **No Unverified Preprocessing:** Any external file must undergo the same rigorous validation checks implemented in `higgs_event_lab.data.validate_raw_chunk`.
* **No Label Contamination:** External sources with different signal/background definitions or event topologies must never be concatenated into the benchmark training set.
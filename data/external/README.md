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

### External Source Record [EXT-001] (Example Template)

* **Dataset / Asset Name:** e.g., UCI HIGGS Checksum & Schema Reference
* **Source URL:** e.g., `https://archive.ics.uci.edu/dataset/280/higgs`
* **Retrieval Date:** `YYYY-MM-DD`
* **License & Attribution:** Open Database License / Creative Commons / Academic Citation
* **SHA-256 Checksum:** `<hex_hash>`
* **Schema Comparison:**
  - Column count: $29$ (1 label + 21 kinematic + 7 invariant masses)
  - Differs from primary UCI source: None (must match `docs/feature_dictionary.csv`)
* **Rationale for Inclusion:** Document why this auxiliary file is necessary.
* **Lead Owner:** Aditya Rana (Dataset Lead)
* **Approval Sign-off:** Krishna (Team Lead) — *Status: Pending / Approved*

---

## 3. Strict Prohibitions

* **No Silent Substitution:** Never silently substitute the primary `data/raw/HIGGS.csv.gz` with an external file.
* **No Unverified Preprocessing:** Any external file must undergo the same rigorous validation checks implemented in `higgs_event_lab.data.validate_raw_chunk`.
* **No Label Contamination:** External sources with different signal/background definitions or event topologies must never be concatenated into the benchmark training set.
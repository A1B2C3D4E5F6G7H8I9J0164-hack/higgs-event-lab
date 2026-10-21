Here is the content formatted with clean spacing, clear headings, and consistent markdown sections for readability:

---

# CHANGELOG

**Change Log**

*Saumya Mishra maintains this file for user-visible changes, documentation updates, known issues, and release notes. Module owners provide concise change details in their PRs.*

> **Note:** This file tracks project history and is not a place to claim model results before a run exists.

---

### Conventions & Guidelines

* **Format:** Follows [Keep a Changelog](https://keepachangelog.com/) conventions (`Added` / `Changed` / `Fixed` / `Removed` / `Known issues`).
* **Dates:** Standard `YYYY-MM-DD` format.
* **Versioning:** Follows Semantic Versioning once the first release is tagged.
* **Adding Entries:** Place entries under `[Unreleased]`, one line per change, including the PR or issue number and the author.
* **Tagging Releases:** When Krishna tags a release, rename `[Unreleased]` to the version and date, then start a new empty `[Unreleased]` section.

---

## [Unreleased]

### Added

* **2026-10-21:** `docs/08_PHYSICS_BACKGROUND.md` — Higgs field, Higgs mechanism, Higgs boson, link to the HIGGS task, glossary (`FND-05`). *(Saumya)*
* **2026-10-21:** `docs/09_DATASET_GUIDE.md` and `docs/feature_dictionary.csv` — Data card and the $21 + 7$ feature dictionary, verified against the UCI page. *(Saumya)*
* **2026-10-21:** `docs/10_OBJECTIVES_AND_METHODOLOGY.md` — Objectives, experiment matrix, pipeline, metrics, and leakage controls. Defaults marked `[PROPOSED]` need Krishna's approval. *(Saumya)*
* **2026-10-21:** `docs/11_SETUP_GUIDE.md` and `requirements.txt` (proposed dependency list). *(Saumya)*
* **2026-10-21:** `docs/12_QA_TEST_PLAN.md` and `team/BUG_TRACKER.md` — QA process, per-owner checklists, and initial bug register. *(Saumya)*
* **2026-10-21:** `tests/unit/test_repo_hygiene.py`, `test_notebook_style.py`, `test_feature_dictionary.py`, and `tests/integration/smoke_check_raw_file.py` — Automated checks for repository hygiene, notebook readability, feature schema, and the raw data file. *(Saumya)*
* **2026-10-21:** `notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb` — Beginner notebook (Markdown-led, no model results). *(Saumya)*
* **2026-10-21:** `reports/01_report_draft_methodology_limitations.md` — Draft report sections for methodology and limitations. No numeric results. *(Saumya)*
* **2026-10-21:** `.gitignore`, `.editorconfig`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/*` — Files promised by repository tree but missing in scaffold (`BUG-001`, `BUG-002`). *(Saumya)*

### Changed

* **2026-10-21:** `CITATION.cff` — Real author names, release date, dataset, and paper references (`BUG-004`). Krishna's surname remains a placeholder. *(Saumya)*
* **2026-10-21:** `docs/07_REFERENCES.md` — Added Higgs-mechanism, discovery, and Nobel references; entries numbered R1 to R8. *(Saumya)*
* **2026-10-21:** `README.md`, `docs/02_REPOSITORY_STRUCTURE.md`, `REPOSITORY_FILE_LIST.txt` — Reading order and file tree updated for new documents. *(Saumya)*
* **2026-10-21:** `team/TASK_BACKLOG.csv` — `FND-05` and `MID-07` moved to *In Review / In Progress*. *(Saumya)*
* **2026-10-21:** Converted `CHANGELOG.md` to Keep a Changelog layout. *(Saumya)*

### Removed

* **2026-10-21:** Stray `.DS_Store` file (macOS metadata) removed and git-ignored (`BUG-003`). *(Saumya)*

### Known Issues

* See `team/BUG_TRACKER.md` for the live list.
* As of this date, no model, loader, dashboard, or deployment code exists; no behavior has been tested beyond documentation and repository hygiene.

---

## [0.1.0-dev] - 2026-10-20

### Added

* Repository foundation (scaffold).
* Project scope, scientific limitations, and non-goals.
* Folder contracts, team ownership, GitHub workflow, and initial task backlog.
* Data-safety rules.
* *Note: No model benchmark or deployed app is claimed at this stage.*

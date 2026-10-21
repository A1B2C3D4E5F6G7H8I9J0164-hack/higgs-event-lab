CHANGELOG
Change Log
Saumya Mishra maintains this file for user-visible changes, documentation updates, known issues and release notes. Module owners provide concise change details in their PRs. This file is project history, not a place to claim model results before a run exists.
Format follows Keep a Changelog conventions (Added / Changed / Fixed / Removed / Known issues). Dates use YYYY-MM-DD. Version numbers follow Semantic Versioning once the first release is tagged.
How to add an entry: put it under [Unreleased], one line per change, with the PR or issue number and the author. When Krishna tags a release, rename [Unreleased] to the version and date, and start a new empty [Unreleased] section.
[Unreleased]
Added
2026-10-21: docs/08_PHYSICS_BACKGROUND.md: Higgs field, Higgs mechanism, Higgs boson, link to the HIGGS task, glossary (FND-05). (Saumya)
2026-10-21: docs/09_DATASET_GUIDE.md and docs/feature_dictionary.csv: data card and the 21 + 7 feature dictionary, verified against the UCI page. (Saumya)
2026-10-21: docs/10_OBJECTIVES_AND_METHODOLOGY.md: objectives, experiment matrix, pipeline, metrics, leakage controls. Defaults marked [PROPOSED] need Krishna's approval. (Saumya)
2026-10-21: docs/11_SETUP_GUIDE.md and requirements.txt (proposed dependency list). (Saumya)
2026-10-21: docs/12_QA_TEST_PLAN.md and team/BUG_TRACKER.md: QA process, per-owner checklists, initial bug register. (Saumya)
2026-10-21: tests/unit/test_repo_hygiene.py, test_notebook_style.py, test_feature_dictionary.py and tests/integration/smoke_check_raw_file.py: automated checks for repository hygiene, notebook readability, feature schema and the raw data file. (Saumya)
2026-10-21: notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb: beginner notebook (Markdown-led, no model results). (Saumya)
2026-10-21: reports/01_report_draft_methodology_limitations.md: draft report sections for methodology and limitations. No numeric results. (Saumya)
2026-10-21: .gitignore, .editorconfig, .github/PULL_REQUEST_TEMPLATE.md, .github/ISSUE_TEMPLATE/*: files that the repository tree promised but the scaffold did not contain (BUG-001, BUG-002). (Saumya)
Changed
2026-10-21: CITATION.cff: real author names, release date, dataset and paper references (BUG-004). Krishna's surname is still a placeholder. (Saumya)
2026-10-21: docs/07_REFERENCES.md: added Higgs-mechanism, discovery and Nobel references; entries numbered R1 to R8. (Saumya)
2026-10-21: README.md, docs/02_REPOSITORY_STRUCTURE.md, REPOSITORY_FILE_LIST.txt: reading order and file tree updated for the new documents. (Saumya)
2026-10-21: team/TASK_BACKLOG.csv: FND-05 and MID-07 moved to In Review / In Progress. (Saumya)
2026-10-21: this file converted to the Keep a Changelog layout. (Saumya)
Removed
2026-10-21: stray .DS_Store file (macOS metadata) removed and git-ignored (BUG-003). (Saumya)
Known issues
See team/BUG_TRACKER.md for the live list. At this date: no model, loader, dashboard or deployment code exists, so no behaviour has been tested beyond documentation and repository hygiene.
[0.1.0-dev] - 2026-10-20: repository foundation (scaffold)
Added
Project scope, scientific limitations and non-goals.
Folder contracts, team ownership, GitHub workflow and initial task backlog.
Data-safety rules.
No model benchmark or deployed app is claimed at this stage.

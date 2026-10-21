# Bug and Issue Tracker

> **Owner:** Saumya Mishra (intake, reproduction, triage, verification). **Fixers:** the module owner (Aditya: data/classical; Rachit: MLP; Ronit: app; Krishna: integration/deployment; Saumya: docs, tests/checklists, packaging, low-risk cross-cutting).
> **Source of truth:** GitHub Issues with label `type:bug`. This file is the readable register that mirrors them and records the first QA sweep. When GitHub Issues exist, put the issue number in the "GH #" column.
> Process rules are in `docs/04_GITHUB_WORKFLOW.md` ("Bug process") and `docs/12_QA_TEST_PLAN.md`.

---

## 1. Severity scale

| Level | Meaning | Examples | Release effect |
|---|---|---|---|
| **P0** | Wrong or misleading results, data leakage, secrets or raw data committed, claim of discovery | Test set used for tuning; label in features; fabricated metric | **Blocks release** (Krishna decides) |
| **P1** | Main workflow broken or cannot be reproduced; schema mismatch; undocumented source | Loader crashes; dashboard cannot read artifacts; missing dependency list | Blocks release unless a workaround is documented |
| **P2** | Defect with a workaround; unclear or misleading wording | Confusing doc, minor plot error | Fix before final submission |
| **P3** | Cosmetic or consistency | Typo, formatting, line endings | Fix when convenient |

## 2. Status values

`New` > `Triaged` (reproduced, severity and owner set) > `In Progress` > `In Review` (fix PR open) > `Verified` (Saumya re-ran the reproduction on a clean clone) > `Closed`.  Also `Won't fix` and `Cannot reproduce` (with a note).

## 3. Triage procedure (Saumya)

1. Acknowledge within the team's agreed response time **[Krishna to set]**.
2. Reproduce from the report. If steps are missing, ask for them using the bug template and mark `Needs info`.
3. Reduce to a minimal failing example (a few lines or a tiny synthetic input).
4. Set severity and assign the module owner. Do not guess at internals of another person's module.
5. Link the fix PR. The fixer adds a regression test or a checklist item.
6. Verify on a clean clone, update `team/CHANGELOG.md`, then close.

---

## 4. Register: first QA sweep, 2026-10-21

Sweep scope: every file in the uploaded scaffold (60 entries), read in full and cross-checked against the other documents and against the official UCI page. No code, notebooks or data existed, so the sweep covers documentation, repository structure and process.

| ID | Sev | Area | Finding | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| BUG-001 | **P0** | Repo / data safety | `.gitignore` and `.editorconfig` were promised by `docs/02` and `REPOSITORY_FILE_LIST.txt` but absent. Without `.gitignore`, `git add .` (the command in `docs/00_START_HERE.md`) would stage raw data, model files and `.env` | Absent from the zip listing | Saumya | **Fixed** 2026-10-21: both files added; `test_gitignore_protects_data_secrets_and_artifacts` guards it |
| BUG-002 | P1 | Process | `.github/PULL_REQUEST_TEMPLATE.md` and `ISSUE_TEMPLATE/*` promised but absent, so the PR and bug process described in `docs/04` had no templates | Absent from the zip listing | Saumya (+ Krishna to review) | **Fixed**: templates added |
| BUG-003 | P3 | Repo hygiene | A macOS `.DS_Store` file was shipped in the archive | `.DS_Store`, 18,436 bytes | Saumya | **Fixed**: removed and git-ignored |
| BUG-004 | P2 | Citation | `CITATION.cff` listed "Project team / Higgs Event Lab" as the author, the version was undocumented elsewhere, and no dataset/paper references were included | `CITATION.cff` (original) | Saumya, Krishna | **Fixed** except Krishna's surname (`TODO-surname`). **Open item for Krishna.** |
| BUG-005 | **P1** | Setup / packaging | No dependency file and no installable package (`src/higgs_event_lab/` has only a README), so the setup guide cannot be fully verified and tests cannot import project code | No `requirements.txt` / `pyproject.toml` | Krishna (package layout), Saumya (docs) | **Open**: proposed `requirements.txt` added; full install untested; pin versions after the first real run |
| BUG-006 | **P1** | Data / sampling | The plan says "begin with 250,000 rows" but does not say how rows are chosen. Taking the first N lines is only valid if the file is not ordered; the exact class fraction is unmeasured | `docs/03`, `docs/05` | Aditya | **Open**: needs an explicit sampling rule and a check comparing first-N versus a random chunk (see `docs/09` section 5.1) |
| BUG-007 | P2 | Scientific wording | `docs/01` described the signal as "Higgs-associated". Per Baldi et al. (2014) the HIGGS signal is a *hypothetical exotic-Higgs* process against a top-quark background, which readers could mistake for the 2012 discovery | `docs/01_PROJECT_CHARTER.md`, original text | Saumya, Krishna review | **Fixed**: wording corrected and cross-referenced to `docs/08` |
| BUG-008 | P1 | Requirements traceability | The course brief PDF is the stated source of requirements but is not in the repository, so no one can verify the stated scope, split or deadlines | `README.md`, `docs/07` | Krishna | **Open** (relates to FND-03): add the PDF to the team drive and link it; copy the deadlines |
| BUG-009 | P3 | Docs consistency | Label list in `docs/00_START_HERE.md` lacked `area:evaluation` and `stage:foundation`, which `docs/04` lists | Diff of the two docs | Saumya | **Fixed** |
| BUG-010 | P3 | Repo hygiene | `team/TASK_BACKLOG.csv` used Windows (CRLF) line endings while the rest of the repo is LF | `file` output on the CSV | Saumya | **Fixed**: converted to LF (also matches `.editorconfig`) |
| BUG-011 | **P1** | Protocol gap | No document fixes split proportions, the random seed, the number of repeated seeds or the target signal efficiency for the operating-point analysis | Searched all docs | Krishna | **Open**: `docs/10` carries `[PROPOSED]` defaults (70/15/15, 3 seeds); Krishna to approve or change in an ADR |
| BUG-012 | P3 | Versioning | `CITATION.cff` said version 0.1.0 while `CHANGELOG.md` said "Unreleased" only | Original files | Saumya | **Fixed**: both now say `0.1.0-dev` |
| BUG-013 | P3 | Docs consistency | `docs/02` and `REPOSITORY_FILE_LIST.txt` did not list the new documents, tests and notebook | n/a | Saumya | **Fixed** (kept in sync by `test_every_file_in_repository_file_list_exists`) |
| BUG-014 | P2 | Data assumptions | Several preprocessing assumptions (b-tag takes few discrete values, heavy-tailed pT/mass features, no missing values, class balance) are expected from physics but unmeasured | `docs/09` section 4 | Aditya | **Open**: confirm in EDA (MID-02) and update the guide with measured facts |

### Summary counts (2026-10-21)

| Severity | Total | Fixed | Open |
|---|---|---|---|
| P0 | 1 | 1 | 0 |
| P1 | 5 | 1 | 4 (BUG-005, 006, 008, 011) |
| P2 | 3 | 2 | 1 (BUG-014) |
| P3 | 5 | 5 | 0 |
| **All** | **14** | **9** | **5** |

(BUG-004 is counted as Fixed with one follow-up item for Krishna.)

---

## 5. Watch list: risks to test as soon as code exists

These are not bugs yet. Each becomes a bug the moment the code fails the check. They are the highest-value regression tests for this project.

| ID | Risk | Test that must exist | Owner of the test |
|---|---|---|---|
| W-01 | Label column leaks into the feature matrix | Assert the feature matrix has 21, 7 or 28 columns and no column equal to the label | Aditya |
| W-02 | Off-by-one in column indices (UCI says "columns 2 to 22", Python starts at 0) | Compare loader column names against `docs/feature_dictionary.csv` | Aditya |
| W-03 | Scaler or imputer fitted before the split | Fit on a toy set where train and test have different means; assert test mean is not used | Aditya, Rachit |
| W-04 | Different models evaluated on different test rows | Assert identical test indices (or hash) across all nine experiments | Krishna |
| W-05 | Threshold or hyperparameters chosen with test data | Assert the operating threshold is computed from validation arrays only | Krishna |
| W-06 | Non-reproducible runs | Same seed gives identical split and near-identical metrics | Rachit, Aditya |
| W-07 | MLP in train mode at inference (dropout and batch-norm active) | Assert `model.eval()` is used and predictions are deterministic | Rachit |
| W-08 | Metric edge cases (one class only, ties) | Unit tests with tiny arrays | Krishna |
| W-09 | Dashboard shows numbers without artifacts | Start the app with an empty `outputs/` and assert an empty-state message | Ronit |
| W-10 | Dashboard accepts malformed or incompatible artifacts | Feed a bad schema and assert a clear error | Ronit |
| W-11 | Accuracy reported alone, or mock data shown unlabelled | Review of the report and dashboard | Saumya |
| W-12 | Text claims discovery or real-world probability | Grep wording in the report and app (see QA plan section 6) | Saumya |

---

## 6. New-bug template (copy into a GitHub issue)

```
Title: [area] short description
Severity: P0 / P1 / P2 / P3
Module / owner:
Steps to reproduce:
Expected:
Actual (paste real output):
Environment (OS, Python, commit):
Minimal failing example:
Regression test or checklist item added:  yes / no
```

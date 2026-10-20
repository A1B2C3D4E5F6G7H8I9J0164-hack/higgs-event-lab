# Contributing to Higgs Event Lab

## Before starting

1. Read `docs/00_START_HERE.md`, `team/TEAM_CHARTER.md` and `docs/04_GITHUB_WORKFLOW.md`.
2. Pick an issue assigned to you. If work is not represented by an issue, create one first.
3. Ask before changing agreed data schemas, output formats, metrics or public interfaces.
4. Keep changes narrow; avoid mixing model work, documentation rewrites and app restyling in one PR.

## Branch and pull request

Create a branch such as `docs/saumya-data-policy`, `data/aditya-loader`, `mlp/rachit-baseline`, `app/ronit-results-view` or `integration/krishna-artifacts`. Open a pull request against `main`, link the issue, explain how you tested it and list docs updated. Do not push directly to `main` after the team starts collaborating.

## Review standard

A reviewer should check scientific correctness, reproducibility, errors, maintainability, data leakage, output compatibility and documentation. The author is responsible for acting on review feedback. Saumya checks that user-facing documentation and the change log reflect the change; module owners remain responsible for code correctness.

## Notebook style

Use Markdown cells as explanation boxes before each major step. State the purpose, method, expected output and interpretation. Keep notebook code cells free from explanatory tutorial comments as agreed by the team. Notebook code should still use clear names and small functions; reusable logic belongs in `src/higgs_event_lab/` rather than being duplicated between notebooks.

## Scientific honesty

Never fabricate results, silently change data sources, tune thresholds on the test set, or describe simulated benchmark performance as a real discovery or detector-ready performance.

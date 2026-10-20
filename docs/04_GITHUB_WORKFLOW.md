# GitHub Workflow for a Five-Person Team

## Repository policy

Start private while the course work is in progress; invite the four teammates by exact GitHub usernames. Public release can be decided later, subject to course rules and checking data/source licenses. Keep `main` as the stable branch.

## Branch names

- `docs/<name>-<topic>`
- `data/aditya-<topic>`
- `classical/aditya-<topic>`
- `mlp/rachit-<topic>`
- `app/ronit-<topic>`
- `integration/krishna-<topic>`
- `fix/<issue-number>-<short-name>`

## Every work item begins with an issue

An issue needs: problem/goal, assigned owner, acceptance criteria, likely files, test plan, dependency/blocker and stage label. Keep tasks small enough to review in one PR where possible.

## Pull request rules

1. Never commit directly to `main` after collaboration begins.
2. A PR links to an issue and summarizes why the change is needed.
3. The author reports test commands and their real result, not merely “tested”.
4. Scientific/model changes list dataset, split, seed and metric effects.
5. App changes include a screenshot or a clear UI behavior description.
6. Documentation and notebook Markdown are updated where behavior changes.
7. A teammate reviews before merge. Authors do not approve their own PRs.
8. Krishna coordinates integration and deployment; Saumya verifies docs/change-log/known-issue updates.
9. Keep changes focused; use squash merges unless the team has a specific reason not to.

## Suggested issue labels

`area:docs`, `area:data`, `area:classical-ml`, `area:deep-learning`, `area:evaluation`, `area:app`, `area:deployment`, `type:bug`, `type:feature`, `type:research`, `priority:p0`, `priority:p1`, `priority:p2`, `stage:foundation`, `stage:midsem`, `stage:endsem`.

## Bug process — Saumya leads

1. Saumya owns the bug queue: reproduce or request reproduction steps, label severity, identify the module owner, record a minimum failing example and track status.
2. Saumya directly fixes documentation, acceptance-test, packaging and low-risk cross-cutting defects when appropriate.
3. For a defect in the data pipeline, classical model, MLP or app implementation, that module's owner fixes the root cause with Saumya coordinating/validating the regression test. This prevents bugs being patched superficially by someone unfamiliar with the module.
4. Saumya updates the change log, usage documentation and known-issues page for user-visible behavior changes.
5. Krishna decides release priority and blocks deployment for P0/P1 defects affecting correctness, leakage or misleading results.

## Commit message pattern

`docs: describe dataset provenance`
`data: validate HIGGS feature schema`
`ml: add logistic regression baseline`
`dl: record MLP training metadata`
`app: show validation-selected operating point`
`fix: reject target leakage in feature loader`

## Review checklist

- [ ] Scope and acceptance criteria met
- [ ] No data leakage or test-set tuning
- [ ] Tests run and results documented
- [ ] No secrets, large data or artifacts added
- [ ] Notebook explanation uses Markdown cells
- [ ] Docs/change log updated (Saumya checks this)
- [ ] Metrics/source/count are accurate
- [ ] Output schema remains compatible
- [ ] Scientific limitations remain visible

# Start Here — Team Leader Setup

## Purpose of this repository

This is the shared home for a five-person course project. It starts with a clear scientific scope, a folder structure, ownership boundaries and a review process. Model implementation follows only after the foundation is agreed.

## Team leader checklist for Krishna

### One-time repository setup

1. Create a new **private** repository named `higgs-event-lab` under the intended GitHub account. Keep it separate from the existing `HiggsLens` repository.
2. Upload the files from this starter scaffold as the initial commit.
3. Add Saumya Mishra, Aditya Rana, Rachit Gupta and Ronit Singh as collaborators using their exact GitHub usernames. Collect the usernames rather than guessing.
4. Keep `main` as the stable branch. Ask everyone to work on issue branches and merge via pull requests.
5. Create labels: `area:docs`, `area:data`, `area:classical-ml`, `area:deep-learning`, `area:app`, `area:deployment`, `type:bug`, `type:feature`, `type:research`, `priority:p0`, `priority:p1`, `priority:p2`, `stage:midsem`, `stage:endsem`.
6. Create a GitHub Project board with columns `Backlog`, `Ready`, `In Progress`, `In Review`, `Blocked`, `Done`.
7. Assign owners using `team/TASK_BACKLOG.csv`; add deadlines after the course's actual dates are confirmed.
8. Agree on a single artifact/output schema before separate model and app modules are developed.

### What not to do yet

- Do not upload the HIGGS dataset or large prediction files.
- Do not start five incompatible notebooks or separate apps.
- Do not add fabricated model scores to screenshots or README badges.
- Do not attempt full-dataset training before the starter experiment is reliable.
- Do not deploy until the test suite, data-source display and limitation statement are in place.

## First team meeting agenda (30–40 minutes)

1. Agree on project goal and scientific boundary (5 min).
2. Read each folder's purpose and confirm ownership (10 min).
3. Agree on repo name, visibility, branch/PR rules and issue board (5 min).
4. Confirm whether mid-semester requires a proposal, demo, notebook or all three (5 min).
5. Create the first five to eight issues and define done criteria (10 min).
6. Record decisions/blockers and next check-in (5 min).

## Definition of done for this setup phase

- A new repository exists at the chosen owner/name and the initial scaffold is committed.
- All five members have collaborator access.
- Every folder contract and team owner is clear.
- No dataset or secret is tracked.
- A starter issue backlog and meeting notes exist.
- Team members can clone the repo and open a small PR.

## Suggested local commands after the empty GitHub repository is created

Unzip the starter, open a terminal in the extracted folder, then run:

```bash
git init
git add .
git commit -m "chore: initialize Higgs Event Lab project scaffold"
git branch -M main
git remote add origin https://github.com/<OWNER>/higgs-event-lab.git
git push -u origin main
```

Replace `<OWNER>` with the intended GitHub username. If the GitHub repository was initialized with its own README or license, reconcile the histories rather than overwriting blindly.

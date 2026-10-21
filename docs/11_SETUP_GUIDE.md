# Setup and Installation Guide

> **Maintainer:** Saumya Mishra (MID-07). **Goal:** a new teammate can get from zero to "tests pass and data checked" without asking anyone.
> **Honest status:** the repository does not yet contain model, loader or dashboard code, so sections 6 to 8 describe what *will* work once those modules land. Each step is tagged **[works now]** or **[pending: owner]**. If a step here stops matching the code, open a bug (see `team/BUG_TRACKER.md`).

---

## 1. What you need

| Requirement | Why | Check |
|---|---|---|
| Git | Clone and make pull requests | `git --version` |
| Python 3.10 or newer (3.12 was used for the QA checks here) | Everything is Python | `python --version` |
| About 4 GB free disk for the compressed dataset, more if you decompress or cache subsets | The HIGGS file is about 2.6 GB compressed | `df -h` |
| 8 GB RAM minimum recommended for 250k-row experiments **[Krishna to confirm]** | Subset-first workflow | n/a |
| A GitHub account that Krishna has invited | Private repository | n/a |
| Optional: an NVIDIA GPU | Faster MLP training; CPU is acceptable for the subset | `nvidia-smi` |

---

## 2. Get the code **[works now]**

```bash
git clone https://github.com/<OWNER>/higgs-event-lab.git
cd higgs-event-lab
git checkout -b docs/<yourname>-<topic>
```

`<OWNER>` is whichever account Krishna created the repository under. Never work directly on `main`.

---

## 3. Create an isolated environment **[works now]**

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

---

## 4. Install dependencies **[works now, with a caveat]**

```bash
pip install -r requirements.txt
```

Caveats (see BUG-005):

- `requirements.txt` is a **proposed** list with lower bounds only. It has not yet been installed as a whole on a clean machine. The first person to do so should record the working versions in the PR.
- `torch` is large (hundreds of MB to a few GB, depending on platform). If you only need the docs and tests, install the light subset instead: `pip install pytest nbformat`.
- If `pip` picks a CUDA build you do not need, follow the selector on pytorch.org for a CPU-only wheel.

Verify the environment:

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch; print('imports ok')"
```

---

## 5. Download and check the data **[works now]**

1. Open the **official** dataset page: <https://archive.ics.uci.edu/dataset/280/higgs>.
2. Download the dataset file (`HIGGS.csv.gz`, about 2.6 GB). If the page offers a zip wrapper, extract the `.csv.gz` from it.
3. Move it to `data/raw/HIGGS.csv.gz`. Do **not** rename the file and do **not** commit it (`.gitignore` blocks it).
4. Record a checksum for the data card:

   ```bash
   shasum -a 256 data/raw/HIGGS.csv.gz        # macOS / Linux
   Get-FileHash data\raw\HIGGS.csv.gz         # Windows
   ```

5. Run the smoke test (streams only the first rows):

   ```bash
   python tests/integration/smoke_check_raw_file.py data/raw/HIGGS.csv.gz --rows 100000
   ```

   Expected: `RESULT: PASS`. A failure lists exactly what is wrong (wrong column count, bad labels, missing values, header row).

6. Copy the printed label counts and b-tag values into your PR or the data card. Report only the numbers the script printed.

**Do not** use `ucimlrepo.fetch_ucirepo(id=280)` or a pandas `read_csv` of the whole file on a laptop; the dataset has about 11 million rows. Chunked loading is Aditya's task (MID-01). Using a third-party mirror is not allowed without Krishna's approval (ADR-0001).

---

## 6. Run the automated checks **[works now]**

```bash
python -m pytest tests/unit -q
```

What this covers today:

| Test file | Checks |
|---|---|
| `test_repo_hygiene.py` | Required files exist, no data/model/secret files, no oversized files, `.gitignore` protections, working Markdown links, backlog format, no unsourced metric numbers in docs |
| `test_notebook_style.py` | Every code cell is preceded by a Markdown explanation, no tutorial comments, short code cells, no large outputs |
| `test_feature_dictionary.py` | The 1 + 21 + 7 column layout matches the UCI order |

Expect all tests to pass on a clean checkout. If one fails, read the assertion message; it names the file and line.

---

## 7. Run the pipeline **[pending: Aditya, Rachit, Krishna]**

When the modules land, this section must be replaced with the real commands. Until then it is a placeholder, and **no results exist**.

| Step | Planned command | Owner |
|---|---|---|
| Prepare a 250k-row subset | `python -m higgs_event_lab.data ...` | Aditya (MID-01) |
| Train classical baselines | `python -m higgs_event_lab.classical ...` | Aditya (MID-03, END-01) |
| Train the MLP | `python -m higgs_event_lab.neural ...` | Rachit (MID-04) |
| Evaluate and write artifacts | `python -m higgs_event_lab.evaluation ...` | Krishna (MID-05) |

Each owner must add the **exact working command** and expected output file names to this table in their PR. Saumya verifies them by running them on a clean clone.

## 8. Run the dashboard **[pending: Ronit]**

```bash
streamlit run app/<main_file>.py
```

Expected today: not applicable. When it exists, with no artifacts present it must show an honest empty-state message and never display placeholder scores.

---

## 9. Notebooks **[works now for the physics notebook]**

```bash
jupyter lab notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb
```

Follow the notebook policy in `notebooks/README.md`: explanations go in Markdown cells, code cells stay short and comment-free, and reusable logic moves to `src/higgs_event_lab/`.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pytest` | Environment not activated or deps not installed | Activate `.venv`, run `pip install -r requirements.txt` |
| `test_no_data_model_or_system_junk_files_are_present` fails | A `.DS_Store`, `.gz`, `.pt` or similar file sits inside the repository folder | Delete it or move it under an ignored path; confirm `.gitignore` is present |
| Smoke test: `File not found` | Wrong path or file name | Check `ls data/raw/` |
| Smoke test: `rows do not have 29 columns` | Truncated download or wrong file | Re-download; compare the checksum |
| Smoke test: `First row looks like a header` | A different edition or a converted file | Stop and tell Krishna; do not silently adapt |
| Out-of-memory while loading | Whole-file read | Use chunked reads (`chunksize`) and a subset |
| Git refuses a push of a large file | Data accidentally staged | `git rm --cached <file>`, confirm `.gitignore`, recommit |
| PowerShell cannot activate the venv | Execution policy | See section 3 |

## 11. Before opening a pull request

- [ ] `python -m pytest tests/unit -q` passes (paste the real output in the PR).
- [ ] No data, model weights, `.env` or notebook outputs with large data are staged (`git status`).
- [ ] Docs and `team/CHANGELOG.md` updated for any user-visible change.
- [ ] You linked the issue and named a reviewer who is not you.

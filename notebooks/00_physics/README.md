# Physics and Problem Context

This folder contains the beginner-level notebook explaining the Higgs field/boson distinction, the simulated classification task, and dataset columns. Keep learning explanations in Markdown cells and cite sources. Do **not** put model-training results here.

## Current Notebooks

| Notebook | Description | Contains Model Results? |
|---|---|---|
| `00_higgs_physics_and_dataset_intro.ipynb` | Markdown-led introduction to the Higgs field, Higgs boson, the HIGGS dataset schema, feature groups, and common pitfalls when reading the file | No |

## Learning Objectives

After working through this notebook, you should be able to:

- Explain the difference between the Higgs field and the Higgs boson in plain language.
- Describe what "signal" and "background" mean in the UCI HIGGS benchmark (exotic-Higgs signal vs. top-pair background).
- List the 21 low-level kinematic features and 7 high-level invariant-mass features.
- Identify common pitfalls when reading the dataset (off-by-one columns, label leakage, fitting on the test set, periodic angles, memory limits).
- Understand the scientific boundary: this is a simulated benchmark, not a real detector analysis.

## Sources

See `docs/07_REFERENCES.md` and `docs/08_PHYSICS_BACKGROUND.md` for citations and further reading.

---

> **Note:** Style is enforced by `tests/unit/test_notebook_style.py`.

---

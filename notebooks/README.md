# Notebook policy

## Planned notebooks

- `00_physics/`: optional beginner-level Higgs physics and dataset interpretation notebook.
- `01_midsem/`: compact MVP, target about 250,000 events when available.
- `02_endsem/`: expanded benchmark, target up to 1,000,000 events when runtime allows.

## Notebook structure

Use Markdown cells as text boxes. Each major section should state:

1. What question this section answers.
2. Why the step is needed.
3. What the code will do.
4. What output to expect.
5. How to interpret it and what could go wrong.

Keep analytical code cells readable and avoid tutorial-style comments in the code. Do not duplicate utility logic that belongs in `src/higgs_event_lab/`. Record source, sample count, seed, split, package versions, model settings and runtime. Do not commit notebook outputs that contain large data or misleading placeholder scores.

# Report Draft: Methodology and Limitations

> **Author:** Saumya Mishra. **Status:** draft v0.1 for team review. **Synthesis owner:** Krishna.
> **Rules for this file:** (1) it contains **no result numbers**; every value that must come from a run is written as `[RUN: ...]` and must be filled from saved artifacts only; (2) text marked **[CONFIRM]** reflects a proposed protocol choice that Krishna must approve; (3) when the final protocol differs from what is written here, edit the text, do not leave stale wording.
> Section numbers follow the report outline in `reports/README.md` (the methods and limitations sections).

---

## Methodology

### M1. Task and research question

We study binary classification of simulated particle-collision events as **signal** or **background**, using the HIGGS benchmark dataset. The research question is: *how do low-level kinematic features, seven expert-derived high-level features, and their combination affect the predictive performance and compute cost of classical machine-learning models versus a compact neural network?* The study is a model-comparison benchmark. It is not a search for a new particle, and its scores are not statistical evidence for any physical claim.

### M2. Data

**Source.** We used the HIGGS dataset from the UCI Machine Learning Repository (Whiteson, 2014; https://doi.org/10.24432/C5V312), released under CC BY 4.0. The dataset was introduced by Baldi, Sadowski and Whiteson (2014). Source file: `[RUN: file name]`, retrieved on `[RUN: date]`, SHA-256 `[RUN: checksum]`.

**Content.** Each record is one Monte Carlo simulated collision event with a class label (1 = signal, 0 = background) and 28 input features: 21 low-level kinematic measurements (the momentum and angles of one charged lepton and up to four jets, jet b-tag values, and the missing transverse energy) and 7 high-level features that are functions of the low-level ones (invariant masses of particle combinations). The complete feature list and order follow the UCI documentation and are stored in `docs/feature_dictionary.csv`. Per the original paper, the signal process involves new, hypothetical Higgs bosons and the background is a top-quark process with the same final-state particles but different kinematics.

**Subset.** The full dataset has about 11 million events. Following the course recommendation, we began with a documented subset of `[RUN: n]` events (target 250,000 for the mid-semester stage, up to 1,000,000 for the end-semester stage). Sampling rule: `[RUN: rule, e.g. stratified random / first N rows, with seed]`. The signal fraction of the subset was `[RUN: measured fraction]`. Rows were read in chunks and the full file was never loaded at once. Data validation (29 columns, valid labels, no missing values) was performed with `tests/integration/smoke_check_raw_file.py`; result: `[RUN: pass/fail and notes]`.

### M3. Feature groups

Three input sets were evaluated: **low-level** (21 features), **high-level** (7 features) and **all** (28 features). The label was always kept separate from the features.

### M4. Data splitting and leakage control **[CONFIRM]**

The subset was split once into stratified training, validation and test partitions in the proportions 70 % / 15 % / 15 % using seed `[RUN: seed]`. The same split indices were used for every model and feature group, so all comparisons use identical test events. The test partition was used only for the final evaluation. The following controls were applied:

- all preprocessing (standardisation, if used) was fitted on the training partition only and then applied to validation and test data;
- hyperparameters, early stopping and (for the operating-point analysis) the decision threshold were chosen on the validation partition only;
- the test partition was evaluated once per final model configuration;
- the label never entered a feature matrix or a scaler, which is checked by automated tests.

### M5. Models

1. **Logistic Regression**: linear baseline; regularisation and solver settings `[RUN: settings]`.
2. **XGBoost**: gradient-boosted decision trees; number of trees, depth, learning rate and early-stopping rounds `[RUN: settings]`.
3. **Compact multilayer perceptron (MLP)**: a fully connected network with batch normalisation and dropout, implemented in PyTorch; layer sizes `[RUN]`, dropout rate `[RUN]`, optimiser and learning rate `[RUN]`, batch size `[RUN]`, early-stopping patience `[RUN]`, device `[RUN]`.

Each model received a documented, comparable tuning budget on the validation partition (`[RUN: e.g. N trials per model]`). Any departure from equal budgets is reported in the results section.

### M6. Evaluation metrics

| Metric | Purpose |
|---|---|
| ROC-AUC | Threshold-independent ranking quality: the probability that a random signal event scores higher than a random background event |
| PR-AUC (average precision) | Precision-recall trade-off for the signal class |
| Brier score and reliability (calibration) curve | Whether predicted scores behave like probabilities on this dataset |
| Training time (and inference time if measured) | Compute cost, recorded as wall-clock seconds with the hardware and thread or device settings |
| Accuracy | Secondary only; not sufficient on its own |

**Operating-point analysis (extension) [CONFIRM].** For a target signal efficiency of `[CONFIRM: value declared in advance]`, we selected the score threshold on the validation partition, froze it, and reported on the test partition the achieved signal efficiency and the background rejection (one minus the false-positive rate). This quantifies a trade-off between keeping signal and removing background. It is **not** a discovery significance and does not include systematic uncertainties.

### M7. Repeats and uncertainty **[CONFIRM]**

Primary experiments were repeated with `[RUN: n]` training seeds on the fixed split, and we report the mean and the spread. Where computed, the test-set ROC-AUC confidence interval was obtained by bootstrap resampling of test events (`[RUN: number of resamples]`). Differences smaller than the observed spread are not described as improvements.

### M8. Reproducibility

Every experiment stores its configuration, data source, row counts, split sizes, seeds, package versions, hardware, timings and metrics in a versioned machine-readable artifact. Tables and figures in this report are generated from those artifacts only. A reproducibility review on a clean clone following `docs/11_SETUP_GUIDE.md` was carried out on `[RUN: date]` with outcome `[RUN: outcome]`. Software versions: `[RUN: Python, numpy, pandas, scikit-learn, xgboost, torch]`. Code and documentation are in the project repository, version `[RUN: tag]`.

### M9. Team workflow and quality assurance

Work was organised through GitHub issues and pull requests, with every change reviewed by a teammate other than its author. A documentation and QA owner maintained a bug register, an automated repository-hygiene and notebook-style test suite, and a release checklist (`docs/12_QA_TEST_PLAN.md`). A release was permitted only if it contained no unsourced metrics, no known leakage, no undocumented data source and no wording that claimed a real discovery.

---

## Limitations

### L1. Simulated data and the nature of the "signal"

- The events are **Monte Carlo simulations**, not recordings from a real detector. Performance on simulated events does not guarantee performance on real collision data, where detector effects, mis-modelling and pile-up differ.
- In this benchmark the signal is a **hypothetical exotic-Higgs process** and the background is a top-quark process (Baldi et al., 2014). The task therefore does not reproduce the 2012 discovery of the 125 GeV Higgs boson, and none of our results bear on whether that particle exists.
- The dataset fixes one physics scenario. A model that separates this signal from this background may not transfer to other signals.

### L2. Class balance and the meaning of scores

- The benchmark contains `[RUN: measured fraction]` signal events, whereas in real collider data the signal is exceedingly rare. Precision, PR-AUC and calibration measured here therefore **do not carry over** to real-world rates.
- Classifier scores are not physical probabilities of an event being a Higgs event. Calibration is assessed only with respect to this benchmark's class mixture.

### L3. Sampling and sample size

- Experiments used a subset of `[RUN: n]` events out of about 11 million. Results may differ at full scale, particularly for the neural network, which can benefit from more data.
- Sampling rule: `[RUN]`. If a non-random rule such as taking leading rows was used, there is a risk that the subset is not representative of the whole file.
- Mid-semester and end-semester runs use different subset sizes and are not directly comparable with each other unless re-run on the same data.
- Our train, validation and test split is our own and differs from the split used in the original paper (if any); our numbers must not be compared with published figures as if the protocols were identical.

### L4. Statistical limitations

- A single data split and a small number of seeds give only a rough estimate of variability. Bootstrap intervals describe test-sample variability, not variation across different training subsets.
- Several metrics and nine feature-model combinations are compared; without a correction for multiple comparisons, small differences may be due to chance.
- The benchmark has no **systematic uncertainties** (detector calibration, theory modelling, background estimation), which dominate real analyses. Without them, no statistical significance can be claimed.

### L5. Model and tuning limitations

- Hyperparameter search was limited by time and compute (`[RUN: budget]`). A model that appears worse may simply be under-tuned, so the comparison reflects "our tuning effort", not each family's ultimate capability.
- Differences between feature groups are confounded with model capacity and preprocessing. For example, a tree model and a linear model react differently to scaling and to heavy-tailed variables.
- The high-level features are deterministic functions of the low-level ones; a model that matches or exceeds high-level performance from low-level inputs shows it can learn those functions, not that the physics features are useless.
- The MLP is intentionally compact. Larger or deeper networks, other architectures, ensembling and feature engineering (for example treating the angle variables as periodic) were not explored.

### L6. Compute and timing

- Training times depend on the hardware, number of threads, GPU use and software versions (`[RUN: hardware]`). They show relative cost on our machine, not universal values.
- Inference time and memory use were `[RUN: measured / not measured]`.

### L7. Operating-point analysis

- The signal-efficiency and background-rejection numbers depend on the chosen target efficiency, the validation partition used to choose the threshold, and the benchmark's class mixture.
- The analysis is a model-comparison tool. It is not a discovery sensitivity estimate.

### L8. Software and process limitations

- Results are tied to the package versions and seeds recorded in the artifacts. Newer library versions, or GPU nondeterminism, can change numbers slightly.
- The test suite checks repository hygiene, schema and process; its coverage of numerical correctness is limited to what module owners implemented. The suite cannot prove the absence of a scientific error.
- The dashboard shows only saved artifacts; it does not retrain models.

### L9. Threats to validity (summary)

| Threat | Direction of possible bias | Mitigation in this study |
|---|---|---|
| Leakage through preprocessing or label | Optimistic | Fit on train only; label separation tests |
| Tuning on the test set | Optimistic | Validation-only selection; test evaluated once |
| Unequal tuning effort across models | Favours the best-tuned model | Documented, comparable budgets |
| Non-representative subset | Either | Sampling rule recorded and checked against another chunk |
| Single seed or split | Unknown noise | Repeated seeds, bootstrap interval |
| Simulation-to-reality gap | Optimistic for real use | Explicit non-claim about real data |

### L10. What the study can and cannot conclude

**It can** compare, under a documented and fair protocol, how well three model families separate simulated signal from background using three input sets, and at what training cost.

**It cannot** establish the existence of any particle, estimate real-data sensitivity, provide calibrated probabilities for real events, or show that one approach is generally superior beyond this benchmark, subset size and tuning budget.

---

## References for these sections

Use the numbered entries in `docs/07_REFERENCES.md`: R2 (dataset), R3 (Baldi et al., 2014) and R1 (course brief).

## Open items before this draft can be finalised

| Item | Owner |
|---|---|
| Approve the split proportions, seed policy and target signal efficiency (BUG-011) | Krishna |
| Fill every `[RUN: ...]` placeholder from saved artifacts | Each module owner for their experiments; Krishna integrates |
| Replace the sampling-rule text with the one actually used (BUG-006) | Aditya |
| Confirm the description of signal and background against Figure 1 of Baldi et al. (2014) | Aditya, Krishna |
| Spot-check R5 to R8 citations against publisher pages | Saumya |

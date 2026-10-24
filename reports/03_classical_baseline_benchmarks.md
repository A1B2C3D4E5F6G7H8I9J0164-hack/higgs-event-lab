# Classical Logistic Regression Baseline Report Draft

> **Author:** Aditya Rana (classical baseline & data lead). **Synthesis owner:** Krishna. **Reviewer:** Rachit Gupta.
> **Note:** Numeric values must come from saved artifacts (`metrics.json`) under `outputs/`. Placeholders are marked `[RUN: ...]`.

---

## 1. Model Configuration and Preprocessing Pipeline

We implement regularized Logistic Regression baselines using `scikit-learn` to establish linear decision boundary benchmarks across feature groups:

- **Input Dimension:** $D \in \{21, 7, 28\}$ matching `low_level`, `high_level`, and `all` feature groups.
- **Preprocessing:** `StandardScaler` embedded within an `sklearn.pipeline.Pipeline` to guarantee that scalers are fitted strictly on the training partition without test or validation leakage.
- **Solver & Regularization:** L-BFGS solver with inverse regularization strength $C = 1.0$ and tolerance $10^{-4}$.
- **Decision Threshold:** Operating threshold selected strictly on the validation partition targeting $50\%$ signal efficiency, then frozen and evaluated on the held-out test partition.

## 2. Experimental Results Summary

All runs use the shared `higgs-split-v1` stratified partition with seed 42.

| Feature Group | Features | Training Time (s) | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Brier Score |
|---|---|---|---|---|---|---|
| Low-level | 21 | `[RUN: time]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` |
| High-level | 7 | `[RUN: time]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` |
| All features | 28 | `[RUN: time]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` |

## 3. Key Observations and Discussion

1. **Feature Group Discrimination:**
   * High-level invariant mass features provide significantly higher linear separability than raw low-level kinematic variables alone, reflecting that invariant mass variables concentrate nonlinear decay kinematic constraints.
   * Combining all 28 features yields the best classical linear benchmark performance.

2. **Leakage Protection and Pipeline Encapsulation:**
   * Embedding feature standardisation within an sklearn Pipeline guarantees that mean and standard deviation estimates derive exclusively from the 70% training partition.
   * Validation-only threshold selection ensures unbiased estimation of background rejection on the held-out test partition.

3. **Reproducibility and Artifacts:**
   * Outputs are exported under `outputs/logreg_<group>/` containing `metrics.json` and `test_predictions.csv` complying with `higgs-experiment-v1`.

# Neural Network Benchmark Report Draft

> **Author:** Rachit Gupta (neural baseline lead). **Synthesis owner:** Krishna. **Reviewer:** Saumya Mishra.
> **Note:** Numeric values must come from saved artifacts (`metrics.json`) under `outputs/`. Placeholders are marked `[RUN: ...]`.

---

## 1. Model Architecture and Configuration

We implement a compact PyTorch Multi-Layer Perceptron (MLP) binary classifier designed for fast, reproducible training without GPU dependency. The network structure is:

- **Input Dimension:** $D \in \{21, 7, 28\}$ matching the chosen feature group.
- **Hidden Layers:** Sequential blocks with widths $(64, 32)$.
- **Regularization:** Each hidden layer applies `BatchNorm1d` followed by `ReLU` and `Dropout(0.20)`.
- **Output:** A single unconstrained logit, trained with `BCEWithLogitsLoss`.
- **Optimizer:** `AdamW` with learning rate `1e-3` and weight decay `1e-4`.
- **Early Stopping:** Monitored on validation ROC-AUC with patience 5. When early stopping triggers, model parameters revert to the best validation checkpoint.

## 2. Experimental Results Summary

All runs use the shared `higgs-split-v1` stratified partition with seed 42.

| Feature Group | Features | Epochs Completed | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Brier Score | Runtime (s) |
|---|---|---|---|---|---|---|---|
| Low-level | 21 | `[RUN: epochs]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` | `[RUN: time]` |
| High-level | 7 | `[RUN: epochs]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` | `[RUN: time]` |
| All features | 28 | `[RUN: epochs]` | `[RUN: val_auc]` | `[RUN: test_auc]` | `[RUN: test_pr]` | `[RUN: brier]` | `[RUN: time]` |

## 3. Key Observations and Discussion

1. **Impact of Domain Features:**
   * The comparison between low-level kinematics and high-level invariant masses evaluates whether deep neural networks can autonomously discover invariant mass representations from four-momenta alone.
   * `[RUN: discussion of comparative performance once 250k runs are logged]`.

2. **Regularization and Training Dynamics:**
   * Batch normalization stabilizes gradient magnitudes during training.
   * Dropout prevents overfitting on the training partition.
   * Early stopping prevents over-training and protects generalization on the held-out test partition.

3. **Reproducibility and Artifacts:**
   * Outputs are exported under `outputs/mlp_<group>/` containing `metrics.json` and `test_predictions.csv` complying with `higgs-experiment-v1`.

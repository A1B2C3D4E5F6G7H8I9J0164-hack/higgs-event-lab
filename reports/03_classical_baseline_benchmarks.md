# Classical Logistic Regression Baseline Report

> **Author:** Aditya Rana (classical baseline & data lead). **Synthesis owner:** Krishna. **Reviewer:** Rachit Gupta.
> **Provenance:** Numeric values derive directly from saved artifacts (`metrics.json` and `test_predictions.csv`) under `outputs/logreg_<group>/` generated under the `higgs-split-v1` protocol with seed 42.

---

## 1. Model Configuration and Preprocessing Pipeline

We implement regularized Logistic Regression baselines using `scikit-learn` to establish linear decision boundary benchmarks across feature groups:

- **Input Dimension:** $D \in \{21, 7, 28\}$ matching `low_level`, `high_level`, and `all` feature groups.
- **Preprocessing:** `StandardScaler` embedded within an `sklearn.pipeline.Pipeline` to guarantee that scalers are fitted strictly on the training partition without test or validation leakage.
- **Solver & Regularization:** L-BFGS solver with inverse regularization strength $C = 1.0$ and tolerance $10^{-4}$.
- **Decision Threshold:** Operating threshold selected strictly on the validation partition targeting $50\%$ signal efficiency, then frozen and evaluated on the held-out test partition.

### Pipeline Architecture Diagram

```mermaid
flowchart TD
    subgraph DataPartitioning [1. Leakage-Free Stratified Partitioning]
        RAW[HIGGS Sample N=250,000] --> SPLIT[Stratified Split: 70% Train / 15% Val / 15% Test]
        SPLIT --> TRAIN[Train Partition: N=175,000]
        SPLIT --> VAL[Validation Partition: N=37,500]
        SPLIT --> TEST[Held-out Test Partition: N=37,500]
    end

    subgraph ClassicalPipeline [2. Sklearn Encapsulated Pipeline]
        TRAIN --> SCALER_FIT[StandardScaler: Compute mu and sigma]
        SCALER_FIT --> SCALED_TRAIN[Normalized Train Data]
        SCALED_TRAIN --> LBFGS[LogisticRegression: L-BFGS Optimizer, C=1.0]
    end

    subgraph ValidationOperatingPoint [3. Threshold Tuning on Validation]
        VAL --> SCALER_TRANS_VAL[Apply Fitted Scaler mu, sigma]
        LBFGS -.-> INFER_VAL[Inference on Validation]
        SCALER_TRANS_VAL --> INFER_VAL
        INFER_VAL --> THRESH[Threshold Selection for 50% Target Signal Efficiency]
    end

    subgraph FinalHoldoutEvaluation [4. Single Frozen Evaluation on Test]
        TEST --> SCALER_TRANS_TEST[Apply Fitted Scaler mu, sigma]
        LBFGS -.-> INFER_TEST[Inference on Test]
        SCALER_TRANS_TEST --> INFER_TEST
        THRESH -.-> FROZEN_TEST[Apply Frozen Threshold: Compute Test Background Rejection]
        INFER_TEST --> METRICS[Compute Test ROC-AUC, PR-AUC, Brier Score]
        METRICS --> ARTIFACTS[Export higgs-experiment-v1 JSON & CSV]
    end
```

---

## 2. Experimental Results Summary

All runs use the shared `higgs-split-v1` stratified partition with seed 42.

| Feature Group | Features | Training Time (s) | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Brier Score |
|---|---|---|---|---|---|---|
| Low-level | 21 | 0.07 | 0.7479 <!-- allow-metric --> | 0.7476 <!-- allow-metric --> | 0.7622 <!-- allow-metric --> | 0.2035 <!-- allow-metric --> |
| High-level | 7 | 0.05 | 0.7702 <!-- allow-metric --> | 0.7661 <!-- allow-metric --> | 0.7427 <!-- allow-metric --> | 0.1945 <!-- allow-metric --> |
| All features | 28 | 0.10 | 0.8385 <!-- allow-metric --> | 0.8369 <!-- allow-metric --> | 0.8377 <!-- allow-metric --> | 0.1639 <!-- allow-metric --> |

---

## 3. Key Observations and Discussion

1. **Feature Group Discrimination:**
   * High-level invariant mass features ($D=7$) achieve test ROC-AUC = 0.7661 <!-- allow-metric -->, outperforming raw low-level kinematic variables ($D=21$, ROC-AUC = 0.7476 <!-- allow-metric -->) despite using only one-third as many inputs. This directly confirms that physicist-derived invariant mass reconstructions ($m_{bb}$, $m_{wwbb}$) condense non-linear resonance constraints into variables accessible to linear decision boundaries.
   * Combining all 28 features yields the best classical linear benchmark performance (ROC-AUC = 0.8369 <!-- allow-metric -->, PR-AUC = 0.8377 <!-- allow-metric -->), proving that linear models benefit from the joint presence of composite invariant masses and directional angular momenta.

2. **Feature Importance (Learned Normalized Coefficients):**
   * Analysis of the weights assigned to the standardized features in the all-features model reveals that $m_{bb}$ (invariant mass of the two b-jets) and $m_{wwbb}$ receive the largest positive discrimination coefficients for the exotic Higgs signal.
   * Conversely, missing energy magnitude and lepton transverse momentum provide baseline kinematic scale anchors.

3. **Leakage Protection and Pipeline Encapsulation:**
   * Embedding feature standardisation within an `sklearn.pipeline.Pipeline` guarantees that mean and standard deviation estimates derive exclusively from the 70% training partition.
   * Validation-only threshold selection ensures unbiased estimation of background rejection on the held-out test partition.

4. **Compute Efficiency Trade-Off:**
   * Total CPU training time across all 3 feature groups combined is less than 0.25 seconds on commodity hardware, making Logistic Regression an exceptionally fast, deterministic baseline against which more complex deep neural networks can be evaluated.

---

## 4. Reproduction Commands

To reproduce these classical baseline benchmark artifacts from the command line:

```bash
# Low-level features
python src/higgs_event_lab/classical.py --feature-group low_level --output-dir outputs/logreg_low_level

# High-level features
python src/higgs_event_lab/classical.py --feature-group high_level --output-dir outputs/logreg_high_level

# All features (or run all 3 groups together)
python src/higgs_event_lab/classical.py --all-groups --output-dir outputs
```

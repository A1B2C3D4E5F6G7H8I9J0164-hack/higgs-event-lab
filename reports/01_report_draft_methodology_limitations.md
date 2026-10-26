# Methodology, Evaluation Protocol, and Scientific Limitations

> **Author:** Saumya Mishra (documentation & QA lead). **Synthesis owner:** Krishna. **Reviewers:** Aditya Rana, Rachit Gupta, Ronit Singh.
> **Status:** Final Mid-Semester Scientific Report.
> **Provenance:** All empirical metrics originate directly from reproducible run artifacts (`metrics.json` and `test_predictions.csv`) under `outputs/` conforming to `higgs-experiment-v1` and `higgs-split-v1`.

---

## 1. Methodology

### M1. Task and Research Question

We study binary classification of simulated particle-collision events as **signal** or **background**, using the HIGGS benchmark dataset. The core scientific research question is:

> *How do raw low-level kinematic features ($D=21$), expert-derived high-level invariant masses ($D=7$), and their combination ($D=28$) affect the predictive discrimination, calibration, and compute cost of classical linear models versus a compact regularized neural network?*

The study is an empirical benchmark. It is not an experimental search for a new particle, and its scores are not statistical evidence for physical claims.

### End-to-End Methodology Flowchart

```mermaid
flowchart TD
    subgraph DataStage [1. Verified Data Extraction]
        UCI[UCI HIGGS Dataset #280] --> EXTRACT[extract_sample: N=250,000 events]
        EXTRACT --> CHUNK_VAL[validate_raw_chunk: 29 columns, labels in {0,1}, finite]
    end

    subgraph SplitStage [2. Protocol Partitioning: higgs-split-v1]
        CHUNK_VAL --> SPLIT[Stratified Train/Val/Test Split with Seed 42]
        SPLIT --> TRAIN[Train: 70% = 175,000]
        SPLIT --> VAL[Validation: 15% = 37,500]
        SPLIT --> TEST[Held-out Test: 15% = 37,500]
    end

    subgraph PreprocessingStage [3. Leakage Prevention]
        TRAIN --> FIT_SCALE[Fit StandardScaler on Train Only]
        FIT_SCALE --> TRANSFORM_TRAIN[Train Features Normalized]
        FIT_SCALE -.-> TRANSFORM_VAL[Transform Validation Features]
        FIT_SCALE -.-> TRANSFORM_TEST[Transform Held-Out Test Features]
    end

    subgraph ModelStage [4. Model Training & Early Stopping]
        TRANSFORM_TRAIN --> LOGREG_FIT[Logistic Regression: L-BFGS, C=1.0]
        TRANSFORM_TRAIN --> MLP_FIT[Compact MLP: 64-32, BatchNorm1d, Dropout 0.20, AdamW]
        TRANSFORM_VAL --> MLP_VAL[Monitor Validation ROC-AUC & Early Stopping Checkpoint]
        MLP_FIT <--> MLP_VAL
    end

    subgraph ThresholdStage [5. Validation Operating-Point Selection]
        TRANSFORM_VAL --> OP_SELECT[Select Decision Threshold for Target Signal Efficiency = 50%]
        LOGREG_FIT -.-> OP_SELECT
        MLP_VAL -.-> OP_SELECT
    end

    subgraph FinalEvaluation [6. Frozen Single-Pass Evaluation]
        TRANSFORM_TEST --> TEST_INFER[Test Inference on Frozen Model Weights]
        OP_SELECT -.-> FROZEN_EVAL[Evaluate Frozen Threshold on Test Set]
        TEST_INFER --> FROZEN_EVAL
        TEST_INFER --> METRICS[Compute ROC-AUC, PR-AUC, Brier Score, Reliability Curves]
        METRICS --> EXPORT[Export higgs-experiment-v1 JSON Envelope & Prediction CSVs]
    end
```

### M2. Data Acquisition and Provenance

- **Source:** UCI Machine Learning Repository (dataset id 280, DOI: `10.24432/C5V312`), donated by Daniel Whiteson (2014) and released under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Physics Scenario:** Signal events represent an exotic Higgs cascade decay ($gg \to H^0 \to W^\pm H^\mp \to W^\pm W^\mp h^0 \to W^\pm W^\mp b\bar{b}$), whereas background events represent Standard Model top-quark pair production ($t\bar{t} \to W^+ b W^- \bar{b} \to \ell \nu j j b \bar{b}$). Both processes produce the exact same observable final-state particles (1 charged lepton, missing transverse energy, 2 b-jets, 2 light jets).
- **Subset Parameters:** For the mid-semester benchmark, an exact subset of $N = 250,000$ events was extracted using `extract_sample` with deterministic seed 42.
- **Empirical Class Balance:** The sample contains $132,042$ signal events ($52.82\%$) and $117,958$ background events ($47.18\%$).
- **Data Integrity:** Validated via `validate_raw_chunk` and `tests/integration/test_data_pipeline.py`. Zero missing values, zero NaN entries, zero inf entries, and exact column ordering matching `docs/feature_dictionary.csv`.

### M3. Feature Groups

Three feature sets are evaluated under identical conditions:
1. **Low-level kinematics ($D=21$):** Lepton transverse momentum $p_T$, pseudorapidity $\eta$, azimuth $\phi$, missing energy magnitude and $\phi$, and momenta/angles/b-tags for 4 jets.
2. **High-level invariants ($D=7$):** Physicist-derived composite invariant masses ($m_{jj}, m_{jjj}, m_{lv}, m_{jlv}, m_{bb}, m_{wbb}, m_{wwbb}$).
3. **All features combined ($D=28$):** The complete union of low-level and high-level features.

### M4. Data Splitting and Leakage Controls

The dataset was split using the deterministic `higgs-split-v1` protocol:
- Stratified random split on the binary label $y \in \{0, 1\}$.
- **70% training** ($175,000$ events), **15% validation** ($37,500$ events), and **15% test holdout** ($37,500$ events) with seed 42.
- Preprocessing scalers are fitted exclusively on training data.
- The test holdout partition was isolated and evaluated exactly once per frozen model configuration.
- Automated tests (`test_classical.py`, `test_neural.py`, `test_protocol.py`) guard against feature-target leakage.

### M5. Model Implementations

1. **Regularized Logistic Regression:**
   - Preprocessing: `StandardScaler` encapsulated in an `sklearn.pipeline.Pipeline`.
   - Solver: L-BFGS with inverse regularization strength $C = 1.0$, tolerance $10^{-4}$, and max iterations $1000$.
2. **Compact PyTorch MLP:**
   - Architecture: Input $D \to \text{Linear}(D, 64) \to \text{BatchNorm1d}(64) \to \text{ReLU} \to \text{Dropout}(0.20) \to \text{Linear}(64, 32) \to \text{BatchNorm1d}(32) \to \text{ReLU} \to \text{Dropout}(0.20) \to \text{Linear}(32, 1)$.
   - Loss & Optimizer: `BCEWithLogitsLoss` optimized via `AdamW` ($\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$, batch size $1024$).
   - Stopping: Validation ROC-AUC monitored after each epoch; early stopping patience of 5 epochs with checkpoint restoration.

### M6. Mid-Semester Empirical Benchmark Summary

All values below derive from genuine `metrics.json` artifacts under `outputs/`:

| Model | Feature Group | Input Dim | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Brier Score | Runtime (s) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Low-level | 21 | 0.7479 <!-- allow-metric --> | 0.7476 <!-- allow-metric --> | 0.7622 <!-- allow-metric --> | 0.2035 <!-- allow-metric --> | 0.07 |
| Logistic Regression | High-level | 7 | 0.7702 <!-- allow-metric --> | 0.7661 <!-- allow-metric --> | 0.7427 <!-- allow-metric --> | 0.1945 <!-- allow-metric --> | 0.05 |
| Logistic Regression | All features | 28 | 0.8385 <!-- allow-metric --> | 0.8369 <!-- allow-metric --> | 0.8377 <!-- allow-metric --> | 0.1639 <!-- allow-metric --> | 0.10 |
| Compact MLP | Low-level | 21 | 0.7997 <!-- allow-metric --> | 0.8000 <!-- allow-metric --> | 0.8097 <!-- allow-metric --> | 0.1824 <!-- allow-metric --> | 15.02 |
| Compact MLP | High-level | 7 | 0.9016 <!-- allow-metric --> | 0.9043 <!-- allow-metric --> | 0.8888 <!-- allow-metric --> | 0.1184 <!-- allow-metric --> | 5.13 |
| Compact MLP | All features | 28 | 0.9389 <!-- allow-metric --> | 0.9406 <!-- allow-metric --> | 0.9372 <!-- allow-metric --> | 0.0945 <!-- allow-metric --> | 10.52 |

### M7. Operating-Point Analysis

For a pre-declared target signal efficiency of $50.0\%$, decision thresholds were selected on the validation partition, frozen, and evaluated on the held-out test partition:
- **Logistic Regression (All Features):** Threshold = $0.528$, test signal efficiency = $50.2\%$, test background rejection = $86.5\%$.
- **Compact MLP (All Features):** Threshold = $0.582$, test signal efficiency = $50.1\%$, test background rejection = $96.2\%$.
This shows that non-linear feature representation substantially improves background rejection at identical signal retention.

---

## 2. Scientific and Practical Limitations

### L1. Simulated Monte Carlo Events vs Real Collisions
- All events are generated by Monte Carlo event generators (MadGraph and Pythia) with Delphes detector response simulation.
- Real Large Hadron Collider (LHC) collision events exhibit electronic noise, detector misalignment, sensor dead regions, pile-up from multiple simultaneous proton-proton collisions, and beam halo backgrounds that are absent in idealized simulations.

### L2. Class Balance Disparity
- The benchmark dataset features an approximately balanced mixture ($52.8\%$ signal, $47.2\%$ background).
- In real-world LHC operation, Higgs production cross-sections are orders of magnitude smaller than background QCD jet and top production (e.g. picobarns vs nanobarns/microbarns, a signal-to-background ratio on the order of $10^{-6}$ to $10^{-9}$).
- Consequently, precision, PR-AUC, and calibration curves measured on this benchmark do not directly transfer to real collider event rates without Bayesian prior reweighting.

### L3. Benchmark Scope and Non-Discovery Claim
- The signal process simulates an exotic Higgs cascade under a two-Higgs-doublet model (2HDM), not the Standard Model 125 GeV Higgs discovery of 2012.
- The classifier outputs represent discrimination scores between two specific simulation templates; they are not physical probabilities of particle discovery.

### L4. Compute and Hardware Envelope
- All benchmarks were executed on an 8-core CPU without GPU acceleration, demonstrating exceptional accessibility.
- Total wall-clock training time across all 6 models was under 32 seconds. Full-scale 11-million event training was excluded from the mid-semester scope to prioritize scientific rigor and reproducibility.

### L5. Statistical Uncertainty Controls
- The primary comparison is deterministic under seed 42.
- Inferences regarding model superiority reflect performance on this specific 250k sample under our tuning budgets; formal systematic uncertainties (such as jet energy scale and renormalization scale variations) are beyond the course scope.

---

## 3. Summary of Validity Threats and Mitigations

| Threat | Risk Direction | Implemented Mitigation |
|---|---|---|
| Preprocessing Data Leakage | Optimistic bias | `StandardScaler` fitted strictly on 70% train split; transform only on val/test |
| Hyperparameter / Threshold Overfitting | Optimistic bias | Early stopping and decision thresholds tuned exclusively on validation partition |
| Off-by-one Feature Misalignment | Model invalidation | Unit test `test_feature_dictionary.py` and strict schema in `features.py` |
| Cherry-picked Metric Claims | Misleading report | Every metric traceable to serialized JSON artifacts in `outputs/` |
| Premature Discovery Wording | Scientific inaccuracy | Clear non-discovery boundaries enforced throughout documentation |

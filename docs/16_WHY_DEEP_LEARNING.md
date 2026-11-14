# Why deep learning for HIGGS, and what we test

The course brief for Project 10 asks for three things: an AML baseline (Logistic Regression and XGBoost), a statement of the gap, and a neural network that is compared fairly on one held-out test set with ROC-AUC, PR-AUC, calibration and training time at a fixed data budget. This note covers the research behind the gap and the extra deep-learning experiments that answer it.

## 1. The research paper

Baldi, Sadowski and Whiteson, *Searching for exotic particles in high-energy physics with deep learning*, Nature Communications 5, 4308 (2014), arXiv:1402.4735. It is the paper the UCI HIGGS dataset comes from.

**The physics problem.** The signal is a hypothetical heavy neutral Higgs boson (H⁰) that decays through a charged Higgs (H±) and W bosons into a light Higgs (h⁰ → bb̄). The background is top-quark pair production (tt̄), which leaves almost the same set of particles in the detector: one lepton, missing energy and four jets, two of them b-tagged. The two processes differ mainly in how those particles' momenta are correlated. Section 4 of `docs/08_PHYSICS_BACKGROUND.md` has the decay chain.

**Two kinds of features.**

- 21 *low-level* features: what the detector measures directly (transverse momentum, η and φ for each lepton and jet, b-tags, missing energy).
- 7 *high-level* features: invariant masses (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb). Physicists compute these from the low-level ones by hand, because they peak at known particle masses for signal events.

**What the paper found** (Table 1, AUC on 500k held-out simulated events, with the networks trained on the rest of the simulated sample):

| Model | Low-level only | High-level only | All 28 |
|---|---:|---:|---:|
| Boosted decision trees | 0.73 | 0.78 | 0.81 |
| Shallow NN (one hidden layer) | 0.733 | 0.777 | 0.816 |
| Deep NN (5 × 300, tanh) | **0.880** | 0.800 | **0.885** |

Three things stand out:

1. Trees and shallow networks need the hand-made invariant masses. Without them they drop to about 0.73.
2. The deep network gets 0.880 **from the low-level features alone**, which is better than any model given the physicists' features. It learns the useful non-linear combinations itself.
3. Adding the high-level features to the deep network barely helps (0.880 to 0.885), because it had already discovered what they encode.

The paper also converts AUC into discovery significance: the deep network on all features reaches 5.0σ, against 3.7σ for the shallow network, on the same simulated sample. In high-energy physics, 5σ is the threshold for claiming a discovery.

**How they trained it.** Five hidden layers of 300 tanh units, learning rate 0.05 decaying every batch, momentum rising from 0.9 to 0.99 over 200 epochs, weight decay 1e-5, mini-batches of 100, and early stopping on a 500k validation set. Each network trained for 200 to 1000 epochs on GPUs. Unsupervised pre-training and extra layers added training time without helping.

## 2. Why our mid-semester MLP tied XGBoost

The mid-semester network had two hidden layers (64 and 32 units, about 4k parameters), saw 175k training events (70% of 250k) and stopped within 30 epochs. In the paper's terms that is a *shallow* network on a *small* sample, which is exactly the case where the paper also found neural nets and trees roughly equal. So the tie was expected, and it isn't evidence against deep learning.

The deep-learning advantage in the paper depends on three things we did not have: depth and width, enough data, and long enough training.

## 3. A second line of evidence: tabular deep learning

Gorishniy, Rubachev, Khrulkov and Babenko, *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 (arXiv:2106.11959), benchmark deep models against gradient-boosted trees on eleven tabular datasets, including a 98k-event HIGGS subset. Their conclusions are relevant here:

- A plain residual MLP (ResNet-style blocks: BatchNorm, Linear, ReLU, Dropout, Linear, plus a skip connection) is a strong baseline that many papers leave out. On HIGGS it beats NODE, a much more complex differentiable-tree model.
- No model wins on every dataset. Gradient boosting stays very strong on small tables, so any deep-learning claim has to be backed by a fair comparison rather than assumed.

That is why our benchmark includes both a Baldi-style deep MLP and a residual MLP, and keeps a properly tuned XGBoost as the opponent.

## 4. What we added

| Change | Where | Why |
|---|---|---|
| `deep` preset: 5 × 300, BatchNorm, ReLU, dropout 0.1, AdamW, cosine learning-rate schedule, up to 60 epochs, patience 8 | `src/higgs_event_lab/neural.py` | Baldi-style depth and width, with modern optimisation so it trains in minutes instead of hundreds of epochs |
| `resnet` preset: 4 residual blocks of width 256 | same | Gorishniy et al. architecture; skip connections keep gradients healthy in deep stacks |
| Whole-tensor mini-batching instead of a per-row `DataLoader` | same | The old loop collated one row at a time and was the main reason training was slow |
| XGBoost: up to 3000 trees, depth 8, early stopping on validation log-loss | `src/higgs_event_lab/classical.py`, release runner | So the AML baseline is tuned too and the comparison is fair, not a weak tree against a big network |
| Duplicate events removed before sampling | `scripts/release_benchmark.py` | The raw file repeats some events verbatim. The first release run found 44 (250k) and 454 (1M) identical events in both train and test |
| Nested budgets 250k ⊂ 1M ⊂ 2.5M, one load of the raw file | same | A learning curve with the same code and seed, which is the "fixed budget before scaling" the brief asks for |
| Expected calibration error next to Brier score | same | The brief asks for calibration |

## 5. Questions the benchmark answers

1. **Fixed budget (the brief's fair comparison).** At each budget, every model sees identical train, validation and test indices. Report ROC-AUC, PR-AUC, Brier, ECE and training time.
2. **Does the gap grow with data?** Compare the XGBoost and deep-network curves across 250k, 1M and 2.5M. The paper predicts the network keeps improving after trees flatten out.
3. **Does the network learn the physics features?** Compare each model on low-level features only against high-level features only. If a network on raw kinematics beats XGBoost on the physicists' invariant masses, it has learned something like those features itself.
4. **Shallow versus deep.** The compact 64-32 MLP stays in the benchmark as the "shallow" control.

Results are in `outputs/release/release_summary.json` and in Section 7.1 of `reports/HiggsLens_Final_Report.md`.

## 6. Limits

- We train on at most 2.5M events, a quarter of the paper's training set, with one seed per budget. Absolute AUCs will be lower than the paper's 0.885, and small differences (under about 0.002) should not be ranked.
- HIGGS is simulated. A better AUC here means a better classifier on this benchmark, not a physics discovery.

## References

1. P. Baldi, P. Sadowski and D. Whiteson. Searching for exotic particles in high-energy physics with deep learning. *Nature Communications* 5, 4308, 2014. arXiv:1402.4735.
2. Y. Gorishniy, I. Rubachev, V. Khrulkov and A. Babenko. Revisiting deep learning models for tabular data. *NeurIPS*, 2021. arXiv:2106.11959.
3. K. He, X. Zhang, S. Ren and J. Sun. Deep residual learning for image recognition. *CVPR*, 2016.
4. I. Loshchilov and F. Hutter. SGDR: stochastic gradient descent with warm restarts. *ICLR*, 2017 (cosine schedule).

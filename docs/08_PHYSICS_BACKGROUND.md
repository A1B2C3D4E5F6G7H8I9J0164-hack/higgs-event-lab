# Physics Background: The Higgs Field, the Higgs Boson and What This Project Does

> **Audience:** a reader with no particle-physics background (course assessors, teammates, viva panel).
> **Maintainer:** Saumya Mishra. **Status:** first full draft, awaiting Krishna's scientific review.
> **Rule:** every physics statement below is either textbook-standard (sources in `docs/07_REFERENCES.md`) or marked *[verify]*.

---

## 1. The big picture in five sentences

1. The Standard Model of particle physics describes the known elementary particles and three of the four fundamental forces.
2. Early versions of the theory had a problem: the force-carrying W and Z bosons *had to* be massless for the maths to work, yet experiments show they are heavy.
3. The **Higgs mechanism** (proposed in 1964) solves this: a field that fills all of space, the **Higgs field**, interacts with these particles and gives them mass without breaking the theory.
4. The **Higgs boson** is a ripple (a quantum excitation) of that field. Finding it was the proof that the field really exists.
5. Our project does **not** study the field itself. We use simulated collision events to compare machine-learning methods that tell "signal" events from "background" events.

---

## 2. The Higgs field

Fields are the fundamental objects in modern physics. An electron is a ripple of the electron field; a photon is a ripple of the electromagnetic field. Most fields are "zero" in empty space, which is why empty space looks empty.

The Higgs field is different. In the lowest-energy state of the universe (the vacuum), its value is **not zero**. It has a constant, non-zero value everywhere, called the *vacuum expectation value* (about 246 GeV in the usual convention *[verify with a textbook/PDG]*).

**Analogy (useful but imperfect):** picture a room filled with a uniform crowd. A particle that does not interact with the crowd moves freely (massless). A particle that interacts strongly with the crowd is slowed down, as if it were heavy. The stronger the interaction, the larger the mass. The analogy fails in several ways (the crowd is not made of particles, and nothing is "dragging" the particle), so use it only as an intuition.

---

## 3. The Higgs mechanism

### 3.1 The problem it solves

The weak force is carried by the W⁺, W⁻ and Z bosons. The theory that unifies the weak and electromagnetic forces (electroweak theory) has a deep symmetry (*gauge symmetry*). Writing an explicit mass term for the W and Z into the equations would break this symmetry and make the theory mathematically inconsistent. Yet experiments in the 1980s measured W and Z masses of roughly 80 and 91 GeV.

### 3.2 The idea: spontaneous symmetry breaking

- The laws of the theory keep the symmetry, but the **lowest-energy state** of the Higgs field does not.
- The Higgs potential energy is often drawn as a "Mexican hat": the top of the hat (field = 0) is unstable, and the field rolls into the circular valley (field ≠ 0).
- Choosing one point in the valley *breaks* the symmetry of the state while the underlying laws stay symmetric. This is called **spontaneous symmetry breaking**.

### 3.3 What it gives mass to

| Particle type | How it gets mass | Notes |
|---|---|---|
| W⁺, W⁻, Z bosons | Interaction with the non-zero Higgs field | Gauge bosons absorb components of the Higgs field and become massive |
| Photon, gluon | Does not couple this way | Stay massless |
| Quarks and charged leptons (e.g. top quark, electron) | Yukawa coupling to the Higgs field | Mass is proportional to coupling strength; the top quark couples most strongly |
| Neutrinos | Not explained by the minimal mechanism | An open question in physics |

### 3.4 Common misconceptions to avoid in the report

- **"The Higgs gives mass to everything."** No. Most of the mass of a proton or neutron comes from the energy of the strong force binding quarks and gluons, not from the Higgs field.
- **"The Higgs boson is what gives mass."** No. The *field* does the work; the boson is the observable ripple that proves the field exists.
- **"This project simulates the Higgs mechanism."** No (see section 6).

### 3.5 History (short, citable)

- **1964:** the mechanism was proposed independently by Englert and Brout, by Higgs, and by Guralnik, Hagen and Kibble (three papers in *Physical Review Letters*, vol. 13).
- **4 July 2012:** the ATLAS and CMS experiments at CERN's Large Hadron Collider (LHC) announced observation of a new particle consistent with the Higgs boson, with a mass of about 125 GeV.
- **2013:** the Nobel Prize in Physics was awarded to François Englert and Peter Higgs.

Full citations are in `docs/07_REFERENCES.md`.

---

## 4. The Higgs boson

- It is a **spin-0** particle (a scalar) with no electric charge. Its mass is about 125 GeV (roughly 130 times a proton's mass).
- It is extremely short-lived (decays in about 10⁻²² seconds *[verify]*), so detectors never see it directly. Physicists infer it from the particles it decays into.
- At the LHC it is rare. Most proton-proton collisions are ordinary "background" processes, and picking out the rare interesting ones is a **classification problem**, which is why machine learning is used.
- A common decay is into a pair of bottom quarks (b b̄). This is why the dataset has *b-tag* inputs and a derived feature called `m_bb`.

---

## 5. How this connects to the HIGGS dataset (important nuance)

The UCI HIGGS dataset comes from the paper of Baldi, Sadowski and Whiteson (2014). According to that paper, the task is:

- **Signal:** a process in which *new, theoretical (exotic) Higgs bosons* are produced (the paper's Figure 1 shows heavy neutral and charged Higgs states, written H⁰ and H±, in a decay chain that ends with a Standard-Model-like Higgs h⁰).
- **Background:** a process involving **top quarks** that produces *the identical decay products* but with different kinematics (the energy and direction patterns of the particles).

Consequences for how we write about it:

1. The dataset's "signal" is **not** a re-creation of the 2012 Higgs discovery. It is a **benchmark problem** built from a hypothetical exotic-Higgs scenario. Do not write "our model detects the Higgs boson".
2. Signal and background end in the **same final-state particles** (a charged lepton, a neutrino seen only as missing energy, and several jets). A model must therefore rely on subtle differences in momenta and angles.
3. All events are **computer-simulated** (Monte Carlo). They are not recordings from the real ATLAS or CMS detectors.

*[verify]* Aditya/Krishna should confirm this wording against Figure 1 and the Methods section of the Nature Communications paper before the final report.

---

## 6. Scientific boundary of the project

| We DO | We do NOT |
|---|---|
| Classify simulated events as signal or background | Simulate the Higgs field or the Higgs mechanism |
| Compare Logistic Regression, XGBoost and a small MLP | Claim discovery of any particle |
| Compare low-level, high-level and all features | Produce detector-grade physics results |
| Report ROC-AUC, PR-AUC, calibration, training time | Convert classifier scores into real-world probabilities of "being a Higgs" |
| Optionally report background rejection at a chosen signal efficiency | Calculate statistical significance (e.g. "5 sigma") |

---

## 7. Glossary (FND-05 deliverable)

| Term | Beginner-friendly meaning |
|---|---|
| **Standard Model** | The current theory of known elementary particles and three of the four forces |
| **Boson / fermion** | Boson: a force-carrier or scalar particle (photon, W, Z, Higgs). Fermion: a matter particle (quarks, leptons) |
| **Higgs field** | A field with non-zero value everywhere; interacting with it gives W, Z and fermions their mass |
| **Higgs boson** | The quantum ripple of the Higgs field; observed in 2012 at about 125 GeV |
| **Symmetry breaking** | Laws are symmetric but the lowest-energy state is not |
| **GeV** | Giga-electron-volt, a unit of energy (and of mass via E = mc²). A proton is about 0.94 GeV |
| **Collider / LHC** | A machine that smashes particles together at high energy; LHC is at CERN |
| **Event** | One recorded (or simulated) collision, described by the particles that came out |
| **Monte Carlo simulation** | Random-sampling computer simulation of physics processes |
| **Signal / background** | Signal: events from the process we want to find. Background: other processes that look similar |
| **Lepton** | A light particle such as an electron or muon (here, one charged lepton per event) |
| **Neutrino** | Neutral, nearly undetectable particle; its presence is inferred from missing energy |
| **Missing (transverse) energy** | Momentum imbalance in the plane perpendicular to the beam, hinting at undetected particles |
| **Jet** | A spray of particles produced by a quark or gluon |
| **b-jet / b-tag** | A jet from a bottom quark; b-tagging is an algorithmic score for how b-like a jet is |
| **Transverse momentum (pT)** | Momentum perpendicular to the beam direction |
| **Pseudorapidity (eta)** | A measure of the angle relative to the beam (large |eta| is near the beam line) |
| **Azimuthal angle (phi)** | The angle around the beam axis (periodic: -π and π are the same direction) |
| **Invariant mass** | The mass computed from the energy and momentum of a group of particles; peaks reveal parent particles |
| **Low-level features** | The 21 direct kinematic measurements |
| **High-level features** | The 7 quantities derived from the low-level ones by physicists |
| **ROC-AUC** | Probability a random signal event is scored higher than a random background event (0.5 = guessing, 1.0 = perfect) |
| **PR-AUC / average precision** | Summary of the precision-recall curve; more informative when classes are imbalanced |
| **Calibration** | Whether a score of 0.8 really means about 80 % of such events are signal |
| **Signal efficiency** | Fraction of true signal events kept after a threshold |
| **Background rejection** | Fraction of background events removed after the same threshold |
| **Data leakage** | Information from the test set (or label) influencing training, giving falsely optimistic results |

---

## 8. Sources used for this document

See `docs/07_REFERENCES.md` (entries R1 to R8). Where a number is marked *[verify]*, replace it with a cited value or delete it before submission.

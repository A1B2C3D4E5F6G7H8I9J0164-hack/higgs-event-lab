# References

Numbering is stable (R1, R2, ...) so that documents can cite "R3". Add new entries at the end.
Entries marked **†** were not re-opened against the publisher or official page while this file was written. They must be verified before the final submission.

## Course brief

**R1.** The course brief for Project 10, "Higgs Boson Signal Detection", defines the scope: simulated signal/background classification; low-level and expert-derived feature experiments; Logistic Regression and XGBoost baselines; a compact MLP with batch normalization and dropout; a stratified subset-first workflow; and comparison using ROC-AUC, PR-AUC, calibration and training time. *(The PDF is not stored in this repository. Krishna to add it to the team drive and link it here.)*

## Dataset and original paper

**R2.** Whiteson, D. (2014). *HIGGS* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5V312 (DOI †). Dataset page: https://archive.ics.uci.edu/dataset/280/higgs. Licence: CC BY 4.0.

**R3.** Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications, 5*, 4308. https://doi.org/10.1038/ncomms5308. Preprint: arXiv:1402.4735.

## Physics background

**R4.** CERN. *The Higgs boson*. https://home.cern/science/physics/higgs-boson/ †

**R5.** Englert, F., & Brout, R. (1964). Broken symmetry and the mass of gauge vector mesons. *Physical Review Letters, 13*, 321 †. Higgs, P. W. (1964). Broken symmetries and the masses of gauge bosons. *Physical Review Letters, 13*, 508 †. Guralnik, G. S., Hagen, C. R., & Kibble, T. W. B. (1964). Global conservation laws and massless particles. *Physical Review Letters, 13*, 585 †.

**R6.** ATLAS Collaboration (2012). Observation of a new particle in the search for the Standard Model Higgs boson with the ATLAS detector at the LHC. *Physics Letters B, 716*, 1–29. https://doi.org/10.1016/j.physletb.2012.08.020 †

**R7.** CMS Collaboration (2012). Observation of a new boson at a mass of 125 GeV with the CMS experiment at the LHC. *Physics Letters B, 716*, 30–61. https://doi.org/10.1016/j.physletb.2012.08.021 †

**R8.** The Nobel Prize in Physics 2013 (Englert and Higgs). https://www.nobelprize.org/prizes/physics/2013/summary/ †

## Models, software and evaluation methods

**R9.** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. https://doi.org/10.1145/2939672.2939785 †

**R10.** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://scikit-learn.org †

**R11.** Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32*. https://pytorch.org †

**R12.** Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *Proceedings of ICLR*. †

**R13.** Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of ICML*. †

**R14.** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*, 1929–1958. †

**R15.** Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. †

**R16.** Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. In *Proceedings of ICML*. †

**R17.** Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. †

**R18.** Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of ICML*. †

**R19.** Streamlit Inc. *Streamlit documentation*. https://docs.streamlit.io †

## Citation rule

Cite the actual dataset source and version used in each experiment. When quoting figures, results or claims from papers or web pages, cite the primary source and separate facts from the team's inference. Model scores in this repository must come from saved runs in `outputs/`, never from published paper scores.

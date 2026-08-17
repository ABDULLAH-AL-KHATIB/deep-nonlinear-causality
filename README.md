# Deep Nonlinear Causality (`deep-nonlinear-causality`)

[![PyPI version](https://badge.fury.io/py/deep-nonlinear-causality.svg)](https://pypi.org/project/deep-nonlinear-causality/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1352--2348-green.svg)](https://orcid.org/0000-0002-1352-2348)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Author:** **Dr. Abdullah Mohammad Ghazi Al khatib**  
**Department:** Department of Banking and Financial Institutions  
**Faculty:** Faculty of Administrative Sciences  
**Affiliation:** **Al-Sham Private University**, Damascus, Syrian Arab Republic  
**Academic Email:** [abduall.kh.foas@aspu.edu.sy](mailto:abduall.kh.foas@aspu.edu.sy)  
**Secondary Email:** [genius.275@hotmail.com](mailto:genius.275@hotmail.com)  
**ORCID iD:** [https://orcid.org/0000-0002-1352-2348](https://orcid.org/0000-0002-1352-2348)  
**GitHub:** [https://github.com/ABDULLAH-AL-KHATIB](https://github.com/ABDULLAH-AL-KHATIB)  
**Release Date:** 17-08-2026  

---

## 🔬 Overview
`deep-nonlinear-causality` is a high-performance scientific framework for discovering **multivariate, non-linear Granger causality and conditional causality** in complex dynamical systems.

Classical Granger causality relies on linear Vector Autoregressions (VAR), failing in systems with non-linear couplings, regime shifts, and threshold effects. This package provides a non-parametric, permutation-based causal inference engine using **Deep Neural Networks** and **Gradient Boosted Decision Trees**.

### Core Capabilities:
- **Multivariate Causal Inputs ($X \in \mathbb{R}^k$):** Test joint causality of multiple coupled variables.
- **Conditional Causality ($X \to Y \mid Z$):** Control for external confounders $Z$ to eliminate spurious causality.
- **Autocorrelation Preservation:** Employs **Circular Block Bootstrap** to maintain serial correlation and avoid inflated Type-I errors.
- **Restricted Model Caching:** Fits the baseline model once per hypothesis test, speeding up computation by ~50%.
- **Memory Engine:** Automatic TensorFlow session clearing and garbage collection to prevent GPU/RAM memory leaks.

---

## 🧠 Supported Estimators
1. **Deep Learning:** LSTM, GRU, 1D-CNN (Temporal Convolutional), MLP.
2. **Machine Learning:** Random Forest, Gradient Boosting, XGBoost, LightGBM, Kernel Ridge.

---

## 📦 Installation

```bash
pip install deep-nonlinear-causality[all]
```

---

## ⚡ Quick Start Examples

### 1. Bivariate Nonlinear Causality Test
```python
import numpy as np
from deepcausality import CausalityTest

# Generate coupled nonlinear series: X causes Y
n = 500
np.random.seed(42)
X = np.random.randn(n, 1)
Y = np.zeros(n)
for t in range(2, n):
    Y[t] = 0.4 * Y[t-1] + 0.35 * (X[t-1, 0]**2) + np.random.normal(0, 0.05)

# Run Causality Test using MLP Neural Network
engine = CausalityTest(random_state=42)
result = engine.run_test(y=Y, x=X, lag=2, model_type='mlp', n_permutations=50)

print(f'Is Causal: {result["is_causal"]}')
print(f'Empirical p-value: {result["p_value"]:.4f}')
print(f'Relative Gain: {result["observed_statistic"]:.4f}')
```

### 2. Multivariate & Conditional Causality Test ($X \to Y \mid Z$)
```python
from deepcausality import ConditionalCausalityTest

X = np.random.randn(n, 3)  # 3 causal candidate series
Z = np.random.randn(n, 1)  # 1 confounding series

cond_engine = ConditionalCausalityTest(random_state=42)
res = cond_engine.run_test(y=Y, x=X, z=Z, lag=2, model_type='lstm', n_permutations=30)

print(f'Conditional Causality Detected: {res["is_causal"]}')
```

---

## 🌍 Research Application Domains
- **Banking, Finance & Econometrics:** Volatility spillovers, systemic risk contagion, crypto-equity linkages.
- **Neuroscience & Biomedicine:** Effective brain connectivity from EEG/fMRI, epileptic seizure onset tracking.
- **Climatology & Earth Sciences:** Ocean-atmosphere teleconnections, emissions vs. climate anomaly couplings.
- **Industrial IoT & Smart Grids:** Fault propagation detection, predictive maintenance in aerospace turbines.
- **Genomics:** Gene Regulatory Network (GRN) reconstruction from longitudinal single-cell RNA-seq.

---

## 📚 Citation
If you use this package in your academic research, please cite it as follows:

```bibtex
@software{Alkhatib_deep_nonlinear_causality_2026,
  author       = {Al khatib, Abdullah Mohammad Ghazi},
  title        = {{deep-nonlinear-causality: Multivariate Nonlinear Granger Causality Detection with Deep Learning and Machine Learning}},
  year         = {2026},
  publisher    = {PyPI},
  organization = {Department of Banking and Financial Institutions, Faculty of Administrative Sciences, Al-Sham Private University},
  address      = {Damascus, Syrian Arab Republic},
  url          = {https://github.com/ABDULLAH-AL-KHATIB/deep-nonlinear-causality},
  orcid        = {https://orcid.org/0000-0002-1352-2348},
  version      = {1.0.2}
}
```

---

## 📜 License
MIT License. Copyright (c) 2026 Dr. Abdullah Mohammad Ghazi Al khatib.

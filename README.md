# Deep Nonlinear Causality (deepcausality)

**Author**: Dr. Abdullah Mohammad Ghazi Al khatib  
**Affiliation**: Al Sham Private University  
**Email**: genius.275@hotmail.com  
**GitHub**: https://github.com/ABDULLAH-AL-KHATIB  
**Release Date**: 17-08-2026  

---

## Overview
A Python framework for discovering multivariate, non-linear Granger causality and conditional causality in complex dynamical systems using Deep Learning (LSTM, GRU, CNN, MLP) and Machine Learning (RF, XGBoost, LightGBM, Kernel Ridge).

## Installation
```bash
pip install deep-nonlinear-causality[all]
```

## Quick Start
```python
import numpy as np
from deepcausality import CausalityTest

n = 500
X = np.random.randn(n, 2)
Y = np.zeros(n)
for t in range(2, n):
    Y[t] = 0.4 * Y[t-1] + 0.3 * (X[t-1, 0]**2) + np.random.normal(0, 0.05)

engine = CausalityTest(random_state=42)
res = engine.run_test(Y, X, lag=2, model_type='mlp', n_permutations=50)
print(f'Is Causal: {res["is_causal"]}, p-value: {res["p_value"]:.4f}')
```

## License
MIT License. Copyright (c) 2026 Dr. Abdullah Mohammad Ghazi Al khatib.

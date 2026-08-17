import pytest
import numpy as np
from deepcausality import CausalityTest
from deepcausality.utils import prepare_data

def test_dimensions():
    y = np.random.randn(100)
    x = np.random.randn(100, 2)
    z = np.random.randn(100, 1)
    X_f, X_r, y_t = prepare_data(y, x, lag=2, z=z)
    assert X_f.shape == (98, 8)
    assert X_r.shape == (98, 4)
    assert y_t.shape == (98,)

def test_pipeline():
    np.random.seed(42)
    n = 150
    x = np.random.randn(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.4 * y[t-1] + 0.3 * (x[t-1]**2) + np.random.normal(0, 0.05)
    engine = CausalityTest(random_state=42)
    res = engine.run_test(y, x, lag=2, model_type="random_forest", n_permutations=3)
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0

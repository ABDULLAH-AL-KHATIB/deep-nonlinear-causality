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

# ---------------- Panel data tests ----------------
def _make_panel(n_entities=8, T=40, causal=True, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros((n_entities, T)); x = np.zeros((n_entities, T))
    for i in range(n_entities):
        e = rng.normal(0, 0.5, T); ex = rng.normal(0, 1, T)
        x[i, 0] = ex[0]
        for t in range(1, T):
            x[i, t] = 0.5 * x[i, t-1] + ex[t]
            y[i, t] = 0.4 * y[i, t-1] + (0.3 * (x[i, t-1] ** 2) if causal else 0.0) + e[t]
    return y, x

def test_to_panel_arrays():
    import pandas as pd
    from deepcausality.utils import to_panel_arrays
    y, x = _make_panel(4, 10)
    rows = [{"id": i, "t": t, "y": y[i, t], "x": x[i, t]} for i in range(4) for t in range(10)]
    df = pd.DataFrame(rows)
    y3, x3, z3, ents, ts = to_panel_arrays(df, "id", "t", ["y", "x"])
    assert y3.shape == (4, 10) and x3.shape == (4, 10) and z3 is None
    assert np.allclose(y3, y) and np.allclose(x3, x)

def test_prepare_panel_data_shapes():
    from deepcausality.utils import prepare_panel_data
    y, x = _make_panel(3, 12)
    X_f, X_r, yt, ent = prepare_panel_data(y, x, lag=2)
    assert X_f.shape == (3 * 10, 4)
    assert X_r.shape == (3 * 10, 2)
    assert yt.shape == (30,) and ent.shape == (30,)

def test_block_permutation_panel():
    from deepcausality.utils import block_permutation_panel
    y, x = _make_panel(5, 20)
    xp = block_permutation_panel(y, x, block_size=None, rng=np.random.default_rng(1))
    assert xp.shape == x.shape
    # entity shuffle preserves the multiset of each entity's values? No—preserves whole trajectories
    assert all(any(np.allclose(xp[i], x[j]) for j in range(5)) for i in range(5))
    xb = block_permutation_panel(y, x, block_size=4, rng=np.random.default_rng(2))
    assert xb.shape == x.shape

def test_panel_pipeline():
    from deepcausality import PanelCausalityTest
    y, x = _make_panel(8, 40, causal=True, seed=42)
    engine = PanelCausalityTest(random_state=42)
    res = engine.run_test(y, x, lag=2, model_type="random_forest", n_permutations=3)
    assert res["data_type"] == "panel"
    assert res["n_entities"] == 8 and res["n_periods"] == 40
    assert 0.0 <= res["p_value"] <= 1.0
    # Strong nonlinear link => large improvement of full over restricted model
    assert res["observed_statistic"] > 0.2
    assert res["mse_full"] < res["mse_restricted"]

def test_panel_block_permutation_detects_causality():
    from deepcausality import PanelCausalityTest
    y, x = _make_panel(8, 40, causal=True, seed=42)
    engine = PanelCausalityTest(random_state=42)
    res = engine.run_test(y, x, lag=2, model_type="random_forest",
                          n_permutations=25, block_size=5)
    assert res["permutation_scheme"].startswith("within_entity_block")
    assert res["p_value"] < 0.05 and res["is_causal"]

def test_panel_null_no_causality():
    from deepcausality import PanelCausalityTest
    y, x = _make_panel(8, 40, causal=False, seed=1)
    engine = PanelCausalityTest(random_state=42)
    res = engine.run_test(y, x, lag=2, model_type="random_forest",
                          n_permutations=9, block_size=5)
    assert not res["is_causal"]

def test_panel_dataframe_pipeline():
    import pandas as pd
    from deepcausality import PanelCausalityTest
    y, x = _make_panel(6, 30, causal=True, seed=7)
    df = pd.DataFrame([{"firm": i, "year": t, "y": y[i, t], "x": x[i, t]}
                       for i in range(6) for t in range(30)])
    engine = PanelCausalityTest(random_state=7)
    res = engine.run_test(df, entity_col="firm", time_col="year", var_cols=["y", "x"],
                          lag=2, model_type="gradient_boosting", n_permutations=3,
                          entity_effects=True)
    assert res["entity_effects"] is True
    assert 0.0 <= res["p_value"] <= 1.0

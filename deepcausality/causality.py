import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from .utils import (
    prepare_data, prepare_sequential_data, block_permutation, to_2d_array,
    to_panel_arrays, prepare_panel_data, block_permutation_panel,
)
from .models import create_model, fit_model, predict_model, clean_backend

class CausalityTest:
    """Multivariate Nonlinear Granger Causality Engine by Dr. Abdullah Mohammad Ghazi Al khatib."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def run_test(self, y, x, z=None, lag=2, model_type="mlp", test_size=0.2, n_permutations=50, block_size=None, **kwargs):
        clean_backend()
        y_arr = to_2d_array(y, "y")
        x_arr = to_2d_array(x, "x")
        z_arr = to_2d_array(z, "z") if z is not None else None
        is_seq = model_type.lower() in ["lstm", "gru", "cnn"]

        if is_seq:
            X_f, X_r, y_t = prepare_sequential_data(y_arr, x_arr, lag=lag, z=z_arr)
            sh_f, sh_r = (X_f.shape[1], X_f.shape[2]), (X_r.shape[1], X_r.shape[2])
        else:
            X_f, X_r, y_t = prepare_data(y_arr, x_arr, lag=lag, z=z_arr)
            sh_f, sh_r = X_f.shape[1], X_r.shape[1]

        n_samples = len(y_t)
        split = int(n_samples * (1.0 - test_size))
        if split < 5 or (n_samples - split) < 5:
            raise ValueError(f"Dataset too small ({n_samples}) for test split {test_size}")

        X_f_tr, X_f_te = X_f[:split], X_f[split:]
        X_r_tr, X_r_te = X_r[:split], X_r[split:]
        y_tr, y_te = y_t[:split], y_t[split:]

        # Cached Restricted Model (Baseline H0)
        m_r = create_model(model_type, sh_r, random_state=self.random_state, **kwargs)
        fit_model(m_r, X_r_tr, y_tr, model_type, **kwargs)
        pred_r = predict_model(m_r, X_r_te, model_type)
        mse_r = mean_squared_error(y_te, pred_r)

        # Full Model (Alternative H1)
        m_f = create_model(model_type, sh_f, random_state=self.random_state, **kwargs)
        fit_model(m_f, X_f_tr, y_tr, model_type, **kwargs)
        pred_f = predict_model(m_f, X_f_te, model_type)
        mse_f = mean_squared_error(y_te, pred_f)
        obs_stat = (mse_r - mse_f) / (mse_r + 1e-12)

        # Permutation Distribution
        p_stats = []
        for i in range(n_permutations):
            x_p = block_permutation(x_arr, block_size=block_size, rng=self.rng)
            if is_seq:
                X_fp, _, _ = prepare_sequential_data(y_arr, x_p, lag=lag, z=z_arr)
            else:
                X_fp, _, _ = prepare_data(y_arr, x_p, lag=lag, z=z_arr)
            m_p = create_model(model_type, sh_f, random_state=self.random_state + i + 1, **kwargs)
            fit_model(m_p, X_fp[:split], y_tr, model_type, **kwargs)
            pred_p = predict_model(m_p, X_fp[split:], model_type)
            stat_p = (mse_r - mean_squared_error(y_te, pred_p)) / (mse_r + 1e-12)
            p_stats.append(stat_p)

        p_stats = np.array(p_stats)
        p_val = (np.sum(p_stats >= obs_stat) + 1.0) / (n_permutations + 1.0)
        clean_backend()

        return {
            "p_value": float(p_val),
            "is_causal": bool(p_val < 0.05),
            "observed_statistic": float(obs_stat),
            "mse_full": float(mse_f),
            "mse_restricted": float(mse_r),
            "model_type": model_type,
            "lag": lag,
            "n_permutations": n_permutations,
            "author": "Dr. Abdullah Mohammad Ghazi Al khatib",
            "department": "Department of Banking and Financial Institutions",
            "faculty": "Faculty of Administrative Sciences",
            "affiliation": "Al-Sham Private University, Damascus, Syrian Arab Republic",
            "academic_email": "abduall.kh.foas@aspu.edu.sy",
            "email": "genius.275@hotmail.com",
            "orcid": "https://orcid.org/0000-0002-1352-2348",
            "github": "https://github.com/ABDULLAH-AL-KHATIB"
        }

class PanelCausalityTest(CausalityTest):
    """Nonlinear Granger Causality for panel (cross-sectional time-series) data.

    Accepts either:
      - a long-format pandas DataFrame plus column names, or
      - 3D numpy arrays of shape (n_entities, n_periods[, n_vars]).

    The lag window for each observation never crosses an entity boundary.
    Permutation inference supports two null schemes:
      - block_size None/<=1 : permute whole entity trajectories (default,
        preserves each entity's internal dynamics),
      - block_size >= 2     : circular block bootstrap within each entity.
    Optionally demean y and x within each entity-time pair (two-way fixed
    effects style) via `entity_effects` / `time_effects`.
    """

    def run_test(self, y, x=None, z=None, entity_col="entity", time_col="time",
                 var_cols=None, lag=2, model_type="mlp", test_size=0.2,
                 n_permutations=50, block_size=None, entity_effects=False,
                 time_effects=False, **kwargs):
        clean_backend()

        # ---- Ingest panel data -------------------------------------------
        if isinstance(y, pd.DataFrame):
            vc = list(var_cols) if var_cols else [c for c in y.columns if c not in (entity_col, time_col)]
            if len(vc) < 2:
                raise ValueError("var_cols must contain at least ['y', 'x'] columns")
            y3d, x3d, z3d, entities, times = to_panel_arrays(y, entity_col, time_col, vc)
        else:
            y3d = np.asarray(y, dtype=np.float32)
            x3d = np.asarray(x, dtype=np.float32)
            if y3d.ndim != 2 and y3d.ndim != 3:
                raise ValueError("Panel y must be 2D (entities x periods) or 3D (entities x periods x vars)")
            z3d = np.asarray(z, dtype=np.float32) if z is not None else None
            entities = list(range(y3d.shape[0]))
            times = list(range(y3d.shape[1]))

        if y3d.ndim == 2:
            y3d = y3d[:, :, None]
        if x3d.ndim == 2:
            x3d = x3d[:, :, None]
        if y3d.shape[:2] != x3d.shape[:2]:
            raise ValueError("Y and X panels must share (n_entities, n_periods)")

        # ---- Optional fixed-effects demeaning ----------------------------
        if entity_effects:
            y3d = y3d - y3d.mean(axis=1, keepdims=True)
            x3d = x3d - x3d.mean(axis=1, keepdims=True)
            if z3d is not None:
                z3d = z3d - z3d.mean(axis=1, keepdims=True)
        if time_effects:
            y3d = y3d - y3d.mean(axis=0, keepdims=True)
            x3d = x3d - x3d.mean(axis=0, keepdims=True)
            if z3d is not None:
                z3d = z3d - z3d.mean(axis=0, keepdims=True)

        is_seq = model_type.lower() in ["lstm", "gru", "cnn"]

        # ---- Feature construction ----------------------------------------
        if is_seq:
            n, T = y3d.shape[0], y3d.shape[1]
            if T <= lag:
                raise ValueError(f"Panel periods {T} must be greater than lag {lag}")
            X_f = np.concatenate([np.concatenate([y3d[i, t-lag:t], x3d[i, t-lag:t]] +
                              ([z3d[i, t-lag:t]] if z3d is not None else []), axis=1)
                              for i in range(n) for t in range(lag, T)])
            X_r = np.concatenate([np.concatenate([y3d[i, t-lag:t]] +
                              ([z3d[i, t-lag:t]] if z3d is not None else []), axis=1)
                              for i in range(n) for t in range(lag, T)])
            y_t = np.array([y3d[i, t, 0] for i in range(n) for t in range(lag, T)], dtype=np.float32)
            ent = np.repeat(np.arange(n), T - lag)
            sh_f, sh_r = (X_f.shape[1], X_f.shape[2]), (X_r.shape[1], X_r.shape[2])
        else:
            X_f, X_r, y_t, ent = prepare_panel_data(y3d, x3d, lag=lag, z3d=z3d)
            sh_f, sh_r = X_f.shape[1], X_r.shape[1]

        # ---- Entity-aware train/test split (never leaks across entities) --
        rng_split = np.random.default_rng(self.random_state)
        perm_ent = rng_split.permutation(len(entities))
        n_tr_ent = max(1, int(len(entities) * (1.0 - test_size)))
        if n_tr_ent == len(entities):
            n_tr_ent -= 1
        tr_mask = np.isin(ent, perm_ent[:n_tr_ent])
        if tr_mask.sum() < 5 or (~tr_mask).sum() < 5:
            raise ValueError(f"Panel too small ({len(y_t)} obs) for entity-wise test split {test_size}")
        X_f_tr, X_f_te = X_f[tr_mask], X_f[~tr_mask]
        X_r_tr, X_r_te = X_r[tr_mask], X_r[~tr_mask]
        y_tr, y_te = y_t[tr_mask], y_t[~tr_mask]

        # ---- Baseline (restricted) and full models ------------------------
        m_r = create_model(model_type, sh_r, random_state=self.random_state, **kwargs)
        fit_model(m_r, X_r_tr, y_tr, model_type, **kwargs)
        mse_r = mean_squared_error(y_te, predict_model(m_r, X_r_te, model_type))

        m_f = create_model(model_type, sh_f, random_state=self.random_state, **kwargs)
        fit_model(m_f, X_f_tr, y_tr, model_type, **kwargs)
        mse_f = mean_squared_error(y_te, predict_model(m_f, X_f_te, model_type))
        obs_stat = (mse_r - mse_f) / (mse_r + 1e-12)

        # ---- Panel permutation null ---------------------------------------
        p_stats = []
        for i in range(n_permutations):
            x_p = block_permutation_panel(y3d, x3d, block_size=block_size, rng=self.rng)
            if is_seq:
                n_, T_ = y3d.shape[0], y3d.shape[1]
                xp3 = x_p if x_p.ndim == 3 else x_p[:, :, None]
                X_fp = np.concatenate([np.concatenate([y3d[j, t-lag:t], xp3[j, t-lag:t]] +
                                  ([z3d[j, t-lag:t]] if z3d is not None else []), axis=1)
                                  for j in range(n_) for t in range(lag, T_)])
            else:
                X_fp, _, _, _ = prepare_panel_data(y3d, x_p, lag=lag, z3d=z3d)
            m_p = create_model(model_type, sh_f, random_state=self.random_state + i + 1, **kwargs)
            fit_model(m_p, X_fp[tr_mask], y_tr, model_type, **kwargs)
            pred_p = predict_model(m_p, X_fp[~tr_mask], model_type)
            p_stats.append((mse_r - mean_squared_error(y_te, pred_p)) / (mse_r + 1e-12))

        p_stats = np.array(p_stats)
        p_val = (np.sum(p_stats >= obs_stat) + 1.0) / (n_permutations + 1.0)
        clean_backend()

        return {
            "p_value": float(p_val),
            "is_causal": bool(p_val < 0.05),
            "observed_statistic": float(obs_stat),
            "mse_full": float(mse_f),
            "mse_restricted": float(mse_r),
            "model_type": model_type,
            "lag": lag,
            "n_permutations": n_permutations,
            "data_type": "panel",
            "n_entities": int(len(entities)),
            "n_periods": int(len(times)),
            "n_observations": int(len(y_t)),
            "permutation_scheme": ("entity_shuffle" if (block_size is None or block_size <= 1)
                                   else f"within_entity_block(size={block_size})"),
            "entity_effects": bool(entity_effects),
            "time_effects": bool(time_effects),
            "author": "Dr. Abdullah Mohammad Ghazi Al khatib",
            "department": "Department of Banking and Financial Institutions",
            "faculty": "Faculty of Administrative Sciences",
            "affiliation": "Al-Sham Private University, Damascus, Syrian Arab Republic",
            "academic_email": "abduall.kh.foas@aspu.edu.sy",
            "email": "genius.275@hotmail.com",
            "orcid": "https://orcid.org/0000-0002-1352-2348",
            "github": "https://github.com/ABDULLAH-AL-KHATIB"
        }


class ConditionalCausalityTest(CausalityTest):
    def run_test(self, y, x, z, lag=2, model_type="mlp", **kwargs):
        if z is None: raise ValueError("Conditioning variable Z is required")
        return super().run_test(y=y, x=x, z=z, lag=lag, model_type=model_type, **kwargs)

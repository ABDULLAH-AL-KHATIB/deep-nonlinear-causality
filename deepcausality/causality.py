import numpy as np
from sklearn.metrics import mean_squared_error
from .utils import prepare_data, prepare_sequential_data, block_permutation, to_2d_array
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

        # Cached Restricted Model
        m_r = create_model(model_type, sh_r, random_state=self.random_state, **kwargs)
        fit_model(m_r, X_r_tr, y_tr, model_type, **kwargs)
        pred_r = predict_model(m_r, X_r_te, model_type)
        mse_r = mean_squared_error(y_te, pred_r)

        # Baseline Full Model
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
            "affiliation": "Al Sham Private University",
            "github": "https://github.com/ABDULLAH-AL-KHATIB"
        }

class ConditionalCausalityTest(CausalityTest):
    def run_test(self, y, x, z, lag=2, model_type="mlp", **kwargs):
        if z is None: raise ValueError("Conditioning variable Z is required")
        return super().run_test(y=y, x=x, z=z, lag=lag, model_type=model_type, **kwargs)

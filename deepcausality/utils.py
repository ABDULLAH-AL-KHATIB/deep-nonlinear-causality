import numpy as np
import pandas as pd

def to_2d_array(arr, name="variable"):
    if arr is None:
        return None
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.to_numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"{name} must be 1D or 2D. Shape received: {arr.shape}")
    return arr

def block_permutation(arr, block_size=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = arr.shape[0]
    if block_size is None or block_size <= 1:
        return rng.permutation(arr)
    n_blocks = int(np.ceil(n / block_size))
    start_indices = rng.integers(0, n, size=n_blocks)
    resampled = []
    for s in start_indices:
        resampled.extend([(s + i) % n for i in range(block_size)])
    return arr[resampled[:n]]

def prepare_data(y, x, lag=1, z=None):
    y_2d = to_2d_array(y, "y")
    x_2d = to_2d_array(x, "x")
    z_2d = to_2d_array(z, "z") if z is not None else None
    n = len(y_2d)
    if len(x_2d) != n:
        raise ValueError("X and Y length mismatch")
    if z_2d is not None and len(z_2d) != n:
        raise ValueError("Z and Y length mismatch")
    if n <= lag:
        raise ValueError(f"Sample size {n} must be greater than lag {lag}")

    X_full, X_restr, y_target = [], [], []
    for t in range(lag, n):
        y_past = y_2d[t - lag:t].flatten()
        x_past = x_2d[t - lag:t].flatten()
        if z_2d is not None:
            z_past = z_2d[t - lag:t].flatten()
            X_full.append(np.concatenate([y_past, x_past, z_past]))
            X_restr.append(np.concatenate([y_past, z_past]))
        else:
            X_full.append(np.concatenate([y_past, x_past]))
            X_restr.append(y_past)
        y_target.append(y_2d[t, 0])

    return np.array(X_full, dtype=np.float32), np.array(X_restr, dtype=np.float32), np.array(y_target, dtype=np.float32)

def prepare_sequential_data(y, x, lag=1, z=None):
    y_2d = to_2d_array(y, "y")
    x_2d = to_2d_array(x, "x")
    z_2d = to_2d_array(z, "z") if z is not None else None
    n = len(y_2d)
    if len(x_2d) != n or (z_2d is not None and len(z_2d) != n):
        raise ValueError("Time series dimension mismatch")
    if n <= lag:
        raise ValueError(f"Sample size {n} must be greater than lag {lag}")

    X_full, X_restr, y_target = [], [], []
    for t in range(lag, n):
        y_p = y_2d[t - lag:t]
        x_p = x_2d[t - lag:t]
        if z_2d is not None:
            z_p = z_2d[t - lag:t]
            X_full.append(np.hstack([y_p, x_p, z_p]))
            X_restr.append(np.hstack([y_p, z_p]))
        else:
            X_full.append(np.hstack([y_p, x_p]))
            X_restr.append(y_p)
        y_target.append(y_2d[t, 0])

    return np.array(X_full, dtype=np.float32), np.array(X_restr, dtype=np.float32), np.array(y_target, dtype=np.float32)

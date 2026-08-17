import gc
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def create_model(model_type, input_shape, **kwargs):
    model_type = model_type.lower()
    seed = kwargs.get("random_state", 42)
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 100), max_depth=kwargs.get("max_depth", None), n_jobs=-1, random_state=seed)
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=kwargs.get("n_estimators", 100), learning_rate=kwargs.get("learning_rate", 0.05), max_depth=kwargs.get("max_depth", 3), random_state=seed)
    elif model_type == "kernel_ridge":
        return KernelRidge(kernel=kwargs.get("kernel", "rbf"), alpha=kwargs.get("alpha", 1.0))
    elif model_type == "xgboost":
        if not XGB_AVAILABLE: raise ImportError("Run: pip install xgboost")
        return XGBRegressor(n_estimators=kwargs.get("n_estimators", 100), learning_rate=kwargs.get("learning_rate", 0.05), verbosity=0, random_state=seed, n_jobs=-1)
    elif model_type == "lightgbm":
        if not LGBM_AVAILABLE: raise ImportError("Run: pip install lightgbm")
        return LGBMRegressor(n_estimators=kwargs.get("n_estimators", 100), learning_rate=kwargs.get("learning_rate", 0.05), random_state=seed, n_jobs=-1, verbose=-1)
    elif model_type in ["mlp", "lstm", "gru", "cnn"]:
        if not TF_AVAILABLE: raise ImportError("Run: pip install tensorflow")
        lr = kwargs.get("learning_rate", 0.002)
        h = kwargs.get("hidden_units", 64)
        dr = kwargs.get("dropout_rate", 0.1)
        m = Sequential()
        if model_type == "mlp":
            dim = input_shape if isinstance(input_shape, int) else input_shape[0]
            m.add(Dense(h, activation="relu", input_shape=(dim,)))
            if dr > 0: m.add(Dropout(dr))
            m.add(Dense(h // 2, activation="relu"))
            m.add(Dense(1))
        elif model_type == "lstm":
            m.add(LSTM(h, activation="tanh", input_shape=input_shape))
            if dr > 0: m.add(Dropout(dr))
            m.add(Dense(1))
        elif model_type == "gru":
            m.add(GRU(h, activation="tanh", input_shape=input_shape))
            if dr > 0: m.add(Dropout(dr))
            m.add(Dense(1))
        elif model_type == "cnn":
            m.add(Conv1D(filters=h, kernel_size=kwargs.get("kernel_size", 2), activation="relu", input_shape=input_shape, padding="same"))
            m.add(GlobalMaxPooling1D())
            m.add(Dense(h // 2, activation="relu"))
            m.add(Dense(1))
        m.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return m
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def fit_model(model, X_train, y_train, model_type, **kwargs):
    model_type = model_type.lower()
    if model_type in ["random_forest", "gradient_boosting", "kernel_ridge", "xgboost", "lightgbm"]:
        model.fit(X_train, y_train)
    else:
        epochs = kwargs.get("epochs", 30)
        batch_size = kwargs.get("batch_size", 32)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def predict_model(model, X_val, model_type):
    model_type = model_type.lower()
    if model_type in ["random_forest", "gradient_boosting", "kernel_ridge", "xgboost", "lightgbm"]:
        return model.predict(X_val).flatten()
    else:
        return model.predict(X_val, verbose=0).flatten()

def clean_backend():
    if TF_AVAILABLE:
        tf.keras.backend.clear_session()
    gc.collect()

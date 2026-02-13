import numpy as np

def inject_mcar(X, missing_rate, random_state=42):
    np.random.seed(random_state)

    X_miss = X.copy()
    mask = np.random.rand(*X_miss.shape) < missing_rate

    X_miss[mask] = np.nan

    return X_miss

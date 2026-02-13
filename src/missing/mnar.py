import numpy as np

def inject_mnar(X, missing_rate, random_state=42):
    """
    Inject MNAR missingness.
    Missingness depends on the value itself.
    """

    rng = np.random.default_rng(random_state)

    X_miss = X.copy()

    # Normalize values
    X_norm = (X - np.nanmin(X)) / (
        np.nanmax(X) - np.nanmin(X) + 1e-8
    )

    # Higher values → more missing
    prob_matrix = missing_rate * X_norm

    mask = rng.random(X_miss.shape) < prob_matrix

    X_miss[mask] = np.nan

    return X_miss

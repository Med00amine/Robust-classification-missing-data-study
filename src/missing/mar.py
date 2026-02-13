import numpy as np

def inject_mar(X, missing_rate, random_state=42):
    """
    Inject MAR missingness.
    Missingness depends on other feature values.
    """

    rng = np.random.default_rng(random_state)

    X_miss = X.copy()

    n_samples, n_features = X.shape

    # Choose random feature as driver
    driver_col = rng.integers(0, n_features)

    # Normalize driver feature
    driver_values = X[:, driver_col]
    driver_values = (driver_values - np.nanmin(driver_values)) / (
        np.nanmax(driver_values) - np.nanmin(driver_values) + 1e-8
    )

    # Probability depends on driver feature
    prob_matrix = missing_rate * driver_values[:, None]

    mask = rng.random(X_miss.shape) < prob_matrix

    X_miss[mask] = np.nan

    return X_miss

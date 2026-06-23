import numpy as np

def generate_synthetic_data(n=200, d=1, noise_std=0.5, seed=42):
    """
    Generate synthetic linear regression data.
    y = X @ w_true + noise
    n: number of samples
    d: input dimension (number of features)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d) / np.sqrt(d)
    noise = rng.standard_normal(n) * noise_std
    y = X @ w_true + noise
    return X, y, w_true

def train_test_split(X, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
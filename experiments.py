import numpy as np
from data import generate_synthetic_data, train_test_split
from models import least_squares, ridge_regression, gradient_descent_ls, mse

#The Peak Finder
def run_double_descent_experiment(
    n_train=150,
    d_values=None,
    noise_std=0.5,
    lam=1e-3,
    seed=42
):
    """
    Fix n_train. Vary dimension d from 1 to 3*n_train
    For each d:
      - Generate data in R^d
      - Fit least squares (closed-form / pseudoinverse)
      - Fit ridge regression
      - Record train and test MSE
    """
    if d_values is None:
        # Sweep d: underparameterized, at threshold, overparameterized
        d_values = list(range(1, 3 * n_train + 1, 5))

    n_total = n_train + 200  # fixed test set size
    rng = np.random.default_rng(seed)

    results = {
        'd': [],
        'ls_train': [], 'ls_test': [],
        'ridge_train': [], 'ridge_test': [],
    }

    for d in d_values:
        X, y, _ = generate_synthetic_data(n=n_total, d=d, noise_std=noise_std, seed=seed)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # --- Least Squares ---
        w_ls = least_squares(X_train, y_train)
        ls_train = mse(X_train, y_train, w_ls)
        ls_test  = mse(X_test,  y_test,  w_ls)

        # --- Ridge ---
        w_r = ridge_regression(X_train, y_train, lam=lam)
        r_train = mse(X_train, y_train, w_r)
        r_test  = mse(X_test,  y_test,  w_r)

        results['d'].append(d)
        results['ls_train'].append(ls_train)
        results['ls_test'].append(ls_test)
        results['ridge_train'].append(r_train)
        results['ridge_test'].append(r_test)

        print(f"d={d:4d} | LS train={ls_train:.4f} test={ls_test:.4f} | "
              f"Ridge train={r_train:.4f} test={r_test:.4f}")

    return results


#The Multi-Universe Tester
def run_noise_experiment(n_train=150, d_values=None, noise_levels=None, seed=42):
    """Extension: vary noise std and observe effect on double descent."""
    if noise_levels is None:
        noise_levels = [0.1, 0.5, 1.0, 2.0]
    if d_values is None:
        d_values = list(range(1, 3 * n_train + 1, 5))

    all_results = {}
    for sigma in noise_levels:
        print(f"\n=== Noise std = {sigma} ===")
        res = run_double_descent_experiment(
            n_train=n_train, d_values=d_values,
            noise_std=sigma, seed=seed
        )
        all_results[sigma] = res
    return all_results


#The Mathematical Audit
def run_gd_vs_closedform(n_train=100, d=50, noise_std=0.5, seed=42):
    """Extension: compare gradient descent vs closed-form least squares."""
    n_total = n_train + 200
    X, y, _ = generate_synthetic_data(n=n_total, d=d, noise_std=noise_std, seed=seed)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Closed-form
    w_cf = least_squares(X_train, y_train)
    cf_train = mse(X_train, y_train, w_cf)
    cf_test  = mse(X_test,  y_test,  w_cf)

    # Gradient descent (tune lr for this scale)
    # Good lr ~ 1 / (2 * max_eigenvalue(X^T X / n))
    eigvals = np.linalg.eigvalsh(X_train.T @ X_train / n_train)
    lr = 1.0 / (2 * eigvals.max() + 1e-8)
    w_gd = gradient_descent_ls(X_train, y_train, lr=lr, n_iter=20000)
    gd_train = mse(X_train, y_train, w_gd)
    gd_test  = mse(X_test,  y_test,  w_gd)

    print(f"Closed-form: train={cf_train:.6f}, test={cf_test:.6f}")
    print(f"Grad descent: train={gd_train:.6f}, test={gd_test:.6f}")
    return {
        'closed_form': (cf_train, cf_test),
        'grad_descent': (gd_train, gd_test)
    }
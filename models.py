import numpy as np

def least_squares(X, y):
    """
    Closed-form least squares solution.
    If d < n: w = (X^T X)^{-1} X^T y  (normal equations)
    If d >= n: use minimum-norm solution w = X^T (X X^T)^{-1} y
    This is the Moore-Penrose pseudoinverse solution.
    """
    n, d = X.shape
    if d <= n:
        # Overdetermined or exactly determined: standard normal equations
        # Add tiny jitter for numerical stability near interpolation threshold
        A = X.T @ X
        b = X.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        # Underdetermined: minimum-norm solution via dual formulation
        # w = X^T (X X^T)^{-1} y
        A = X @ X.T
        try:
            alpha = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(X @ X.T, y, rcond=None)[0]
        w = X.T @ alpha
    return w

def ridge_regression(X, y, lam=1e-3):
    """
    Ridge: w = (X^T X + lambda * I)^{-1} X^T y
    Works for all d.
    """
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w

def gradient_descent_ls(X, y, lr=1e-3, n_iter=10000, tol=1e-8):
    """
    Gradient descent for least squares (extension).
    Gradient: 2/n * X^T (X w - y)
    """
    n, d = X.shape
    w = np.zeros(d)
    prev_loss = np.inf
    for _ in range(n_iter):
        grad = (2 / n) * X.T @ (X @ w - y)
        w = w - lr * grad
        loss = np.mean((X @ w - y) ** 2)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return w

def mse(X, y, w):
    """Mean squared error."""
    residuals = X @ w - y
    return np.mean(residuals ** 2)
import numpy as np

#The Double Descent Trigger
def least_squares(X, y):
    """
    Closed-form least squares solution using pseudo-inverse.
    Ensures the interpolation peak naturally forms at d = n.
    """
    n, d = X.shape
    if d < n:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    else:
        return X.T @ np.linalg.pinv(X @ X.T) @ y

#The Smooth Stabilizer
def ridge_regression(X, y, lam=1e-3):
    """
    Ridge regression using primal/dual forms.
    Switches to dual form when d > n to prevent numerical explosion.
    """
    n, d = X.shape
    if d <= n:
        # Primal form: stable when features <= samples
        A = X.T @ X + lam * np.eye(d)
        return np.linalg.solve(A, X.T @ y)
    else:
        # Dual form: stable when features > samples
        A = X @ X.T + lam * np.eye(n)
        return X.T @ np.linalg.solve(A, y)

#The Step-by-Step Learner
def gradient_descent_ls(X, y, lr=1e-3, n_iter=10000, tol=1e-8):
    """
    Gradient descent for least squares (extension)
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

#The Performance Grade
def mse(X, y, w):
    """Mean squared error"""
    residuals = X @ w - y
    return np.mean(residuals ** 2)
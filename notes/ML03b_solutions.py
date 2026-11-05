"""Solutions for notebooks/ML03b_logistic_regression_practice.ipynb.

Try the exercises yourself first. Each block below replaces one "# YOUR CODE HERE"
cell in the notebook: paste the whole block over that cell and run the test cell after it.
The blocks use names the notebook sets up earlier (np, plt, DCAPB, INFL, train, test,
y_train, y_test), so they only run inside the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt


# %% [exercise-1a] sigmoid
def sigmoid(z):
    """The sigmoid g(z) = 1 / (1 + e^(-z)). Works on a number or a whole array."""
    g = 1 / (1 + np.exp(-z))
    return g


# %% [exercise-1b] cost
def compute_cost_logistic(X, y, w, b, lambda_=0.0):
    """Logistic cost: average log loss plus the regularization term λ/(2m)·Σw²."""
    m, n = X.shape
    f = sigmoid(X @ w + b)                                     # predicted chance of "weak"
    f = np.clip(f, 1e-15, 1 - 1e-15)                           # so log(0) never happens
    loss = -y * np.log(f) - (1 - y) * np.log(1 - f)            # one loss per episode
    total_cost = loss.mean() + lambda_ / (2 * m) * np.sum(w ** 2)
    return total_cost


# %% [exercise-1c] gradient
def compute_gradient_logistic(X, y, w, b, lambda_=0.0):
    """Gradient of the cost. Returns dj_db, dj_dw (the same order as the course labs)."""
    m, n = X.shape
    err = sigmoid(X @ w + b) - y                               # f(x) - y for every episode
    dj_dw = (X.T @ err) / m + (lambda_ / m) * w                # b is not regularized
    dj_db = err.mean()
    return dj_db, dj_dw


# %% [exercise-1d] gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.0):
    """Repeat: compute the gradient, take a step of size alpha, record the cost."""
    w, b = w_in.copy(), b_in
    J_history = []
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost_logistic(X, y, w, b, lambda_))
    return w, b, J_history


# %% [exercise-2a] boundary line
def boundary_inflation(dcapb, w, b):
    """Inflation on the decision boundary for each ΔCAPB value.

    The boundary is where w[0]·ΔCAPB + w[1]·inflation + b = 0, so the model gives exactly 50 %.
    """
    inflation = -(b + w[0] * dcapb) / w[1]
    return inflation


# %% [exercise-2b] plot
def plot_decision_boundary(data, w, b):
    """Scatter the episodes (weak / not weak) and draw the boundary line. Returns the axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    weak = data["weak"] == 1
    ax.scatter(data.loc[~weak, DCAPB], data.loc[~weak, INFL], marker="o", label="not weak", alpha=0.7)
    ax.scatter(data.loc[weak, DCAPB], data.loc[weak, INFL], marker="x", label="weak", alpha=0.9)
    xs = np.linspace(data[DCAPB].min(), data[DCAPB].max(), 100)
    ax.plot(xs, boundary_inflation(xs, w, b), color="black", label="decision boundary (50 %)")
    ax.set_ylim(data[INFL].min() - 3, data[INFL].max() + 3)   # keep the view on the episodes
    ax.set_xlabel("ΔCAPB at t (pp of GDP)")
    ax.set_ylabel("inflation at t−1 (%, capped at ±30)")
    ax.legend()
    return ax


# %% [exercise-3a] polynomial features and z-scores
def poly_features(x, degree):
    """Columns x, x², ..., x^degree for a 1-D array x."""
    X_poly = np.column_stack([x ** d for d in range(1, degree + 1)])
    return X_poly

def fit_zscore(X):
    """Mean and standard deviation of every column (ddof = 0, like scikit-learn's StandardScaler)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return mu, sigma

def zscore(X, mu, sigma):
    return (X - mu) / sigma


# %% [exercise-3b] training vs hidden log loss
def train_vs_hidden(degree, lambdas, alpha, num_iters):
    """For each λ: fit on the training countries, return {λ: (training log loss, hidden log loss)}."""
    X_tr = poly_features(train[DCAPB].to_numpy(), degree)
    X_te = poly_features(test[DCAPB].to_numpy(), degree)
    mu, sigma = fit_zscore(X_tr)                               # training countries only
    X_tr, X_te = zscore(X_tr, mu, sigma), zscore(X_te, mu, sigma)
    results = {}
    for lam in lambdas:
        w, b, _ = gradient_descent(X_tr, y_train, np.zeros(degree), 0.0, alpha, num_iters, lam)
        results[lam] = (compute_cost_logistic(X_tr, y_train, w, b, 0.0),    # plain log loss, no λ term
                        compute_cost_logistic(X_te, y_test, w, b, 0.0))
    return results

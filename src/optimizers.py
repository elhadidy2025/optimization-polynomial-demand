"""Optimization algorithms implemented with NumPy only."""

import time
import numpy as np
from objectives import mse_loss, ridge_loss, ridge_grad, lasso_objective, prox_l1_unpenalized_intercept


def _record(history, X, y, w, objective, start_time, iteration, grad_evals=None):
    history["iteration"].append(iteration)
    history["loss"].append(float(objective(X, y, w)))
    history["time"].append(time.perf_counter() - start_time)
    if grad_evals is not None:
        history["grad_evals"].append(int(grad_evals))


def gradient_descent(X, y, objective, gradient, step, max_iter=500, w0=None, record_every=1):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    history = {"iteration": [], "loss": [], "time": []}
    start = time.perf_counter()
    for k in range(max_iter + 1):
        if k % record_every == 0:
            _record(history, X, y, w, objective, start, k)
        if k == max_iter:
            break
        w -= step * gradient(X, y, w)
    return w, history


def nesterov(X, y, objective, gradient, step, max_iter=500, w0=None, record_every=1):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    z = w.copy()
    t = 1.0
    history = {"iteration": [], "loss": [], "time": []}
    start = time.perf_counter()
    for k in range(max_iter + 1):
        if k % record_every == 0:
            _record(history, X, y, w, objective, start, k)
        if k == max_iter:
            break
        w_next = z - step * gradient(X, y, z)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w, t = w_next, t_next
    return w, history


def heavy_ball(X, y, objective, gradient, step, beta=0.8, max_iter=500, w0=None, record_every=1):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    prev = w.copy()
    history = {"iteration": [], "loss": [], "time": []}
    start = time.perf_counter()
    for k in range(max_iter + 1):
        if k % record_every == 0:
            _record(history, X, y, w, objective, start, k)
        if k == max_iter:
            break
        grad = gradient(X, y, w)
        new = w - step * grad + beta * (w - prev)
        prev, w = w, new
    return w, history


def proximal_gradient_l1(X, y, lam=1e-3, step=1e-2, max_iter=500, w0=None, accelerated=False, record_every=1):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    z = w.copy()
    t = 1.0
    history = {"iteration": [], "loss": [], "time": []}
    start = time.perf_counter()
    for k in range(max_iter + 1):
        if k % record_every == 0:
            _record(history, X, y, w, lambda A, b, ww: lasso_objective(A, b, ww, lam), start, k)
        if k == max_iter:
            break
        base = z if accelerated else w
        grad = (X.T @ (X @ base - y)) / X.shape[0]
        w_next = prox_l1_unpenalized_intercept(base - step * grad, step * lam)
        if accelerated:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = w_next + ((t - 1.0) / t_next) * (w_next - w)
            t = t_next
        w = w_next
    return w, history


def mini_batch_indices(n, batch_size, rng):
    return rng.choice(n, size=batch_size, replace=False)


def sgd_ridge(X, y, lam=1e-3, step0=1e-2, batch_size=32, epochs=30, w0=None, decay=0.0, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    n = X.shape[0]
    history = {"iteration": [], "loss": [], "time": [], "grad_evals": []}
    start = time.perf_counter()
    grad_evals = 0
    total_steps = int(epochs * np.ceil(n / batch_size))
    for k in range(total_steps + 1):
        if k % max(1, total_steps // 200) == 0:
            _record(history, X, y, w, lambda A, b, ww: ridge_loss(A, b, ww, lam), start, k, grad_evals)
        if k == total_steps:
            break
        idx = mini_batch_indices(n, batch_size, rng)
        Xb, yb = X[idx], y[idx]
        step = step0 / (1.0 + decay * k)
        grad = ridge_grad(Xb, yb, w, lam)
        w -= step * grad
        grad_evals += len(idx)
    return w, history


def adagrad_ridge(X, y, lam=1e-3, step=0.1, batch_size=32, epochs=30, eps=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    acc = np.zeros_like(w)
    n = X.shape[0]
    history = {"iteration": [], "loss": [], "time": [], "grad_evals": []}
    start = time.perf_counter()
    grad_evals = 0
    total_steps = int(epochs * np.ceil(n / batch_size))
    for k in range(total_steps + 1):
        if k % max(1, total_steps // 200) == 0:
            _record(history, X, y, w, lambda A, b, ww: ridge_loss(A, b, ww, lam), start, k, grad_evals)
        if k == total_steps:
            break
        idx = mini_batch_indices(n, batch_size, rng)
        g = ridge_grad(X[idx], y[idx], w, lam)
        acc += g * g
        w -= step * g / (np.sqrt(acc) + eps)
        grad_evals += len(idx)
    return w, history


def adam_ridge(X, y, lam=1e-3, step=0.01, batch_size=32, epochs=30, beta1=0.9, beta2=0.999, eps=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    n = X.shape[0]
    history = {"iteration": [], "loss": [], "time": [], "grad_evals": []}
    start = time.perf_counter()
    grad_evals = 0
    total_steps = int(epochs * np.ceil(n / batch_size))
    for k in range(total_steps + 1):
        if k % max(1, total_steps // 200) == 0:
            _record(history, X, y, w, lambda A, b, ww: ridge_loss(A, b, ww, lam), start, k, grad_evals)
        if k == total_steps:
            break
        idx = mini_batch_indices(n, batch_size, rng)
        g = ridge_grad(X[idx], y[idx], w, lam)
        step_t = k + 1
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** step_t)
        v_hat = v / (1.0 - beta2 ** step_t)
        w -= step * m_hat / (np.sqrt(v_hat) + eps)
        grad_evals += len(idx)
    return w, history


def saga_ridge(X, y, lam=1e-3, step=1e-2, epochs=30, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    table = np.zeros((n, d))
    table_mean = np.zeros(d)
    history = {"iteration": [], "loss": [], "time": [], "grad_evals": []}
    start = time.perf_counter()
    total_steps = epochs * n
    for k in range(total_steps + 1):
        if k % max(1, total_steps // 200) == 0:
            _record(history, X, y, w, lambda A, b, ww: ridge_loss(A, b, ww, lam), start, k, k)
        if k == total_steps:
            break
        i = rng.integers(0, n)
        xi = X[i]
        gi = (xi @ w - y[i]) * xi
        mask_w = w.copy()
        mask_w[0] = 0.0
        grad_est = gi - table[i] + table_mean + lam * mask_w
        w -= step * grad_est
        table_mean += (gi - table[i]) / n
        table[i] = gi
    return w, history

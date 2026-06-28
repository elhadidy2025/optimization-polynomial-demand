"""Objective functions for polynomial regression."""

import numpy as np


def regularization_mask(d):
    mask = np.ones(d)
    mask[0] = 0.0  # do not regularize intercept
    return mask


def mse_loss(X, y, w):
    r = X @ w - y
    return 0.5 * float(np.mean(r ** 2))


def mse_grad(X, y, w):
    n = X.shape[0]
    return (X.T @ (X @ w - y)) / n


def ridge_loss(X, y, w, lam=1e-3):
    mask = regularization_mask(len(w))
    return mse_loss(X, y, w) + 0.5 * lam * float(np.sum((mask * w) ** 2))


def ridge_grad(X, y, w, lam=1e-3):
    mask = regularization_mask(len(w))
    return mse_grad(X, y, w) + lam * mask * w


def lasso_objective(X, y, w, lam=1e-3):
    mask = regularization_mask(len(w))
    return mse_loss(X, y, w) + lam * float(np.sum(np.abs(mask * w)))


def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def prox_l1_unpenalized_intercept(z, tau):
    out = z.copy()
    out[1:] = soft_threshold(out[1:], tau)
    return out


def nonconvex_l0_smooth_loss(X, y, w, lam=1e-3, tau=0.1):
    mask = regularization_mask(len(w))
    z = mask * w
    penalty = np.sum(1.0 - np.exp(-(z ** 2) / (2.0 * tau ** 2)))
    return mse_loss(X, y, w) + lam * float(penalty)


def nonconvex_l0_smooth_grad(X, y, w, lam=1e-3, tau=0.1):
    mask = regularization_mask(len(w))
    z = mask * w
    grad_pen = lam * np.exp(-(z ** 2) / (2.0 * tau ** 2)) * z / (tau ** 2)
    return mse_grad(X, y, w) + grad_pen


def lipschitz_constant_mse(X, lam=0.0):
    # For 1/(2n)||Xw-y||^2 + lam/2||w||^2, L = ||X||_2^2/n + lam.
    smax = np.linalg.svd(X, compute_uv=False)[0]
    return float((smax ** 2) / X.shape[0] + lam)


def closed_form_ridge(X, y, lam=1e-3):
    d = X.shape[1]
    mask = regularization_mask(d)
    A = (X.T @ X) / X.shape[0] + lam * np.diag(mask)
    b = (X.T @ y) / X.shape[0]
    # Small jitter protects against singular unregularized cases.
    return np.linalg.solve(A + 1e-10 * np.eye(d), b)

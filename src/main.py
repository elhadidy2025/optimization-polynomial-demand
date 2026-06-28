"""Run all experiments for the final project.

Usage from the project root:
    python src/main.py

Only NumPy, pandas, and matplotlib are used for computation and plots.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from data_utils import prepare_design, rmse_original_units, mae_original_units, make_dirs
from objectives import (
    mse_loss,
    mse_grad,
    ridge_loss,
    ridge_grad,
    lasso_objective,
    nonconvex_l0_smooth_loss,
    nonconvex_l0_smooth_grad,
    lipschitz_constant_mse,
    closed_form_ridge,
)
from optimizers import (
    gradient_descent,
    nesterov,
    heavy_ball,
    proximal_gradient_l1,
    sgd_ridge,
    adagrad_ridge,
    adam_ridge,
    saga_ridge,
)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_model(name, w, payload, objective_name="ridge", lam=0.0):
    Xtr, ytr = payload["X_train"], payload["y_train"]
    Xte, yte = payload["X_test"], payload["y_test"]
    y_mean, y_std = payload["y_mean"], payload["y_std"]
    pred_tr = Xtr @ w
    pred_te = Xte @ w
    return {
        "model": name,
        "objective": objective_name,
        "train_rmse": rmse_original_units(ytr, pred_tr, y_mean, y_std),
        "test_rmse": rmse_original_units(yte, pred_te, y_mean, y_std),
        "train_mae": mae_original_units(ytr, pred_tr, y_mean, y_std),
        "test_mae": mae_original_units(yte, pred_te, y_mean, y_std),
        "train_loss_scaled": float(mse_loss(Xtr, ytr, w)),
        "n_features": Xtr.shape[1],
        "lambda": lam,
    }


def save_history_csv(histories, out_path):
    rows = []
    for name, hist in histories.items():
        for i in range(len(hist["iteration"])):
            rows.append({
                "method": name,
                "iteration": hist["iteration"][i],
                "loss": hist["loss"][i],
                "time": hist["time"][i],
                "grad_evals": hist.get("grad_evals", [np.nan] * len(hist["iteration"]))[i],
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_histories(histories, out_path, title, y_label="Training objective"):
    plt.figure(figsize=(7.0, 4.2))
    for name, hist in histories.items():
        plt.plot(hist["iteration"], hist["loss"], label=name, linewidth=1.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar(df, x_col, y_col, out_path, title, xlabel, ylabel):
    plt.figure(figsize=(7.8, 4.5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_algorithm_comparison(base_dir, payload, degree=3, lam=1e-3):
    X, y = payload["X_train"], payload["y_train"]
    L = lipschitz_constant_mse(X, lam)
    obj = lambda A, b, w: ridge_loss(A, b, w, lam)
    grad = lambda A, b, w: ridge_grad(A, b, w, lam)

    histories = {}
    models = {}

    max_iter = 500
    models["Gradient Descent"], histories["Gradient Descent"] = gradient_descent(
        X, y, obj, grad, step=0.85 / L, max_iter=max_iter, record_every=2
    )
    models["Nesterov"], histories["Nesterov"] = nesterov(
        X, y, obj, grad, step=0.95 / L, max_iter=max_iter, record_every=2
    )
    models["Heavy Ball"], histories["Heavy Ball"] = heavy_ball(
        X, y, obj, grad, step=0.55 / L, beta=0.82, max_iter=max_iter, record_every=2
    )

    # Stochastic methods. Steps were chosen conservatively after scaling features and target.
    models["SGD"], histories["SGD"] = sgd_ridge(
        X, y, lam=lam, step0=0.008, batch_size=32, epochs=25, decay=2e-4, seed=1
    )
    models["SAGA"], histories["SAGA"] = saga_ridge(
        X, y, lam=lam, step=1e-4, epochs=25, seed=2
    )
    models["Adagrad"], histories["Adagrad"] = adagrad_ridge(
        X, y, lam=lam, step=0.08, batch_size=32, epochs=25, seed=3
    )
    models["Adam"], histories["Adam"] = adam_ridge(
        X, y, lam=lam, step=0.01, batch_size=32, epochs=25, seed=4
    )

    rows = [evaluate_model(name, w, payload, objective_name="ridge", lam=lam) for name, w in models.items()]
    comp_df = pd.DataFrame(rows).sort_values("test_rmse")
    comp_df.to_csv(os.path.join(base_dir, "results", "algorithm_comparison.csv"), index=False)
    save_history_csv(histories, os.path.join(base_dir, "results", "convergence_histories.csv"))
    plot_histories(
        histories,
        os.path.join(base_dir, "figures", "fig1_algorithm_convergence.png"),
        f"Convergence comparison, degree {degree}, ridge lambda={lam}",
    )
    plot_bar(
        comp_df,
        "model",
        "test_rmse",
        os.path.join(base_dir, "figures", "fig2_test_rmse_by_algorithm.png"),
        "Prediction accuracy by optimizer",
        "Optimizer",
        "Test RMSE (sales units)",
    )
    return comp_df, histories, models


def run_objective_comparison(base_dir, payload, lam=1e-3):
    X, y = payload["X_train"], payload["y_train"]
    L_mse = lipschitz_constant_mse(X, 0.0)
    L_ridge = lipschitz_constant_mse(X, lam)

    results = []
    histories = {}

    w_mse, h_mse = gradient_descent(
        X, y, lambda A, b, w: mse_loss(A, b, w), lambda A, b, w: mse_grad(A, b, w),
        step=0.85 / L_mse, max_iter=500, record_every=2
    )
    histories["MSE convex smooth"] = h_mse
    results.append(evaluate_model("GD", w_mse, payload, objective_name="MSE", lam=0.0))

    w_ridge, h_ridge = nesterov(
        X, y, lambda A, b, w: ridge_loss(A, b, w, lam), lambda A, b, w: ridge_grad(A, b, w, lam),
        step=0.95 / L_ridge, max_iter=500, record_every=2
    )
    histories["MSE + L2 strongly convex"] = h_ridge
    results.append(evaluate_model("Nesterov", w_ridge, payload, objective_name="MSE + L2", lam=lam))

    w_l1, h_l1 = proximal_gradient_l1(
        X, y, lam=lam, step=0.95 / L_mse, max_iter=500, accelerated=True, record_every=2
    )
    histories["MSE + L1 nonsmooth"] = h_l1
    results.append(evaluate_model("FISTA", w_l1, payload, objective_name="MSE + L1", lam=lam))
    results[-1]["sparsity_percent"] = 100.0 * np.mean(np.abs(w_l1[1:]) < 1e-8)

    # Smooth nonconvex L0-like penalty optimized by Adam with full gradients.
    nconv_lam = lam
    tau = 0.12
    obj_nc = lambda A, b, w: nonconvex_l0_smooth_loss(A, b, w, nconv_lam, tau)
    grad_nc = lambda A, b, w: nonconvex_l0_smooth_grad(A, b, w, nconv_lam, tau)
    # Use a small step; the penalty can be locally steep near zero.
    w_nc, h_nc = gradient_descent(X, y, obj_nc, grad_nc, step=0.01, max_iter=500, record_every=2)
    histories["MSE + smooth L0-like nonconvex"] = h_nc
    results.append(evaluate_model("GD", w_nc, payload, objective_name="MSE + smooth L0-like", lam=nconv_lam))
    results[-1]["sparsity_percent"] = 100.0 * np.mean(np.abs(w_nc[1:]) < 1e-4)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, "results", "objective_comparison.csv"), index=False)
    plot_histories(
        histories,
        os.path.join(base_dir, "figures", "fig3_objective_convergence.png"),
        "Convergence for different objective structures",
    )
    plot_bar(
        df.sort_values("test_rmse"),
        "objective",
        "test_rmse",
        os.path.join(base_dir, "figures", "fig4_objective_test_rmse.png"),
        "Objective structure vs prediction accuracy",
        "Objective",
        "Test RMSE (sales units)",
    )
    return df


def run_degree_effect(base_dir, train_csv, store=1, item=1, lam=1e-3):
    rows = []
    for degree in [1, 2, 3, 4, 5]:
        payload = prepare_design(train_csv, store=store, item=item, degree=degree)
        X, y = payload["X_train"], payload["y_train"]
        w = closed_form_ridge(X, y, lam=lam)
        row = evaluate_model(f"degree {degree}", w, payload, objective_name="ridge", lam=lam)
        row["degree"] = degree
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_dir, "results", "degree_effect.csv"), index=False)
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(df["degree"], df["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(df["degree"], df["test_rmse"], marker="o", label="Test RMSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE (sales units)")
    plt.title("Effect of polynomial degree")
    plt.xticks(df["degree"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "figures", "fig5_degree_effect.png"), dpi=180)
    plt.close()
    return df


def run_step_size_effect(base_dir, payload, lam=1e-3):
    X, y = payload["X_train"], payload["y_train"]
    L = lipschitz_constant_mse(X, lam)
    obj = lambda A, b, w: ridge_loss(A, b, w, lam)
    grad = lambda A, b, w: ridge_grad(A, b, w, lam)
    histories = {}
    rows = []
    for mult in [0.25, 0.5, 0.9, 1.1]:
        w, h = gradient_descent(X, y, obj, grad, step=mult / L, max_iter=250, record_every=1)
        name = f"{mult}/L"
        histories[name] = h
        row = evaluate_model(name, w, payload, objective_name="ridge step", lam=lam)
        row["step_multiplier"] = mult
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_dir, "results", "step_size_effect.csv"), index=False)
    plot_histories(histories, os.path.join(base_dir, "figures", "fig6_step_size_effect.png"), "Effect of step size on gradient descent")
    return df


def run_regularization_effect(base_dir, payload):
    rows = []
    X, y = payload["X_train"], payload["y_train"]
    for lam in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        w = closed_form_ridge(X, y, lam=lam)
        row = evaluate_model(f"lambda={lam:g}", w, payload, objective_name="ridge", lam=lam)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_dir, "results", "regularization_effect.csv"), index=False)
    plt.figure(figsize=(7.0, 4.2))
    plt.semilogx(df["lambda"].replace(0.0, 1e-7), df["train_rmse"], marker="o", label="Train RMSE")
    plt.semilogx(df["lambda"].replace(0.0, 1e-7), df["test_rmse"], marker="o", label="Test RMSE")
    plt.xlabel("L2 regularization lambda; zero shown at 1e-7")
    plt.ylabel("RMSE (sales units)")
    plt.title("Effect of regularization parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "figures", "fig7_regularization_effect.png"), dpi=180)
    plt.close()
    return df


def save_prediction_plot(base_dir, best_w, payload):
    dates = pd.to_datetime(payload["dates_test"])
    y_true = payload["y_test"] * payload["y_std"] + payload["y_mean"]
    y_pred = (payload["X_test"] @ best_w) * payload["y_std"] + payload["y_mean"]
    plt.figure(figsize=(8.0, 4.2))
    plt.plot(dates, y_true, label="Actual", linewidth=1.2)
    plt.plot(dates, y_pred, label="Predicted", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Daily sales")
    plt.title("2017 test-set demand: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "figures", "fig8_actual_vs_predicted.png"), dpi=180)
    plt.close()


def main():
    base_dir = project_root()
    make_dirs(base_dir)
    train_csv = os.path.join(base_dir, "data", "train.csv")

    store, item = 1, 1
    degree = 3
    lam = 1e-3
    payload = prepare_design(train_csv, store=store, item=item, degree=degree)

    metadata = {
        "dataset_rows_full": int(pd.read_csv(train_csv, usecols=["sales"]).shape[0]),
        "store": store,
        "item": item,
        "degree_main": degree,
        "main_lambda": lam,
        "n_train_after_lags": int(payload["X_train"].shape[0]),
        "n_test_after_lags": int(payload["X_test"].shape[0]),
        "n_features_degree_main": int(payload["X_train"].shape[1]),
        "raw_feature_columns": ", ".join(payload["feature_cols"]),
    }

    print("Running algorithm comparison...")
    algo_df, histories, models = run_algorithm_comparison(base_dir, payload, degree=degree, lam=lam)

    print("Running objective comparison...")
    objective_df = run_objective_comparison(base_dir, payload, lam=lam)

    print("Running degree effect...")
    degree_df = run_degree_effect(base_dir, train_csv, store=store, item=item, lam=lam)

    print("Running step size effect...")
    step_df = run_step_size_effect(base_dir, payload, lam=lam)

    print("Running regularization effect...")
    reg_df = run_regularization_effect(base_dir, payload)

    best_name = algo_df.iloc[0]["model"]
    best_w = models[best_name]
    save_prediction_plot(base_dir, best_w, payload)

    metadata.update({
        "best_optimizer": str(best_name),
        "best_test_rmse": float(algo_df.iloc[0]["test_rmse"]),
        "best_test_mae": float(algo_df.iloc[0]["test_mae"]),
        "best_degree_by_test_rmse": int(degree_df.sort_values("test_rmse").iloc[0]["degree"]),
        "best_lambda_by_test_rmse": float(reg_df.sort_values("test_rmse").iloc[0]["lambda"]),
    })
    pd.Series(metadata).to_csv(os.path.join(base_dir, "results", "metadata.csv"), header=False)

    print("Done. Results written to figures/ and results/.")
    print(algo_df)


if __name__ == "__main__":
    main()

## Data loading and preprocessing for the demand forecasting experiment

import os
import numpy as np
import pandas as pd
from polynomial_features import PolynomialFeatures


class Standardizer:
    ##Column-wise standardization with zero-variance protection

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_store_item(path, store=1, item=1):
    df = pd.read_csv(path)
    required = {"date", "store", "item", "sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[(df["store"] == store) & (df["item"] == item)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_time_series_features(df):
    ## Create calendar and historical demand features.

    ## Lag and rolling mean are computed from past sales values. Rows without
    ## complete lag information are dropped.
   
    data = df.copy()
    data["t"] = np.arange(len(data), dtype=float)
    data["day_of_week"] = data["date"].dt.dayofweek.astype(float)
    data["month"] = data["date"].dt.month.astype(float)
    data["day_of_year"] = data["date"].dt.dayofyear.astype(float)
    data["weekend"] = (data["day_of_week"] >= 5).astype(float)
    data["lag_1"] = data["sales"].shift(1)
    data["lag_7"] = data["sales"].shift(7)
    data["rolling_mean_7"] = data["sales"].shift(1).rolling(7).mean()
    data = data.dropna().reset_index(drop=True)
    feature_cols = ["t", "day_of_week", "month", "day_of_year", "weekend", "lag_1", "lag_7", "rolling_mean_7"]
    return data, feature_cols


def chronological_split(data, feature_cols, test_year=2017):
    train_mask = data["date"].dt.year < test_year
    test_mask = data["date"].dt.year == test_year
    X_train_raw = data.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train_raw = data.loc[train_mask, "sales"].to_numpy(dtype=float)
    X_test_raw = data.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test_raw = data.loc[test_mask, "sales"].to_numpy(dtype=float)
    dates_train = data.loc[train_mask, "date"].to_numpy()
    dates_test = data.loc[test_mask, "date"].to_numpy()
    return X_train_raw, y_train_raw, X_test_raw, y_test_raw, dates_train, dates_test


def prepare_design(train_csv, store=1, item=1, degree=3, test_year=2017):
    df = load_store_item(train_csv, store=store, item=item)
    data, feature_cols = make_time_series_features(df)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, dates_train, dates_test = chronological_split(
        data, feature_cols, test_year=test_year
    )

    x_scaler = Standardizer()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()
    if y_std == 0:
        y_std = 1.0
    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train = poly.fit_transform(X_train_scaled)
    X_test = poly.transform(X_test_scaled)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "dates_train": dates_train,
        "dates_test": dates_test,
        "feature_cols": feature_cols,
        "poly": poly,
        "x_scaler": x_scaler,
        "y_mean": y_mean,
        "y_std": y_std,
        "data": data,
    }


def rmse_original_units(y_true_scaled, y_pred_scaled, y_mean, y_std):
    y_true = y_true_scaled * y_std + y_mean
    y_pred = y_pred_scaled * y_std + y_mean
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_original_units(y_true_scaled, y_pred_scaled, y_mean, y_std):
    y_true = y_true_scaled * y_std + y_mean
    y_pred = y_pred_scaled * y_std + y_mean
    return float(np.mean(np.abs(y_true - y_pred)))


def make_dirs(base_dir):
    for name in ["figures", "results", "report"]:
        os.makedirs(os.path.join(base_dir, name), exist_ok=True)

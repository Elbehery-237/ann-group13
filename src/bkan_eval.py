"""Evaluation for the B-spline KAN experiments. Owner: Abdelrahman Elbehery.

Implements the evaluation protocol agreed for all four models, so results are comparable:
    Every metric is computed on the TEST split, in ORIGINAL units (float64),
    after undoing the target standardisation used for training.

Why train on standardised targets at all?  SRSD Medium targets span ~1e-37 … ~2e38.
In float32, squared errors on 1e-31 targets underflow to 0 (a model that always
outputs 0 gets MSE = 0.0) and squared errors on 1e38 targets overflow to inf.
Standardising y is only a change of units for training; it is undone here before
anything is reported, so it cannot make a result look better.

The five reported metrics (n test points, e_i = yhat_i - y_i, ybar = mean of y_test):
    MSE   = (1/n) sum e_i^2                       original units squared
    RMSE  = sqrt(MSE)                             original units
    MAE   = (1/n) sum |e_i|                       original units
    R^2   = 1 - MSE / Var(y_test)                 unit-free, 1 = perfect, 0 = predicting ybar
    NRMSE = RMSE / std(y_test)  (ddof = 0)        unit-free, 0 = perfect, 1 = predicting ybar
NRMSE follows Petersen et al., "Deep Symbolic Regression" (ICLR 2021); NRMSE = sqrt(1 - R^2).

Aggregation rule: MSE, RMSE and MAE carry the units of each equation (targets range from
~1e-37 to ~2e38), so they are reported PER EQUATION only. Across equations only the
unit-free NRMSE (headline) and R^2 are summarised (median and IQR).
"""
from __future__ import annotations

import os
import time

import numpy as np

METRICS = ("mse", "rmse", "mae", "r2", "nrmse")


def regression_metrics(y_true, y_pred) -> dict:
    """All metrics in float64, original units. y_true, y_pred: 1-D array-likes of equal length."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    finite = bool(np.all(np.isfinite(y_pred)))
    err = y_pred - y_true
    mse = float(np.mean(err ** 2)) if finite else float("inf")
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err))) if finite else float("inf")
    std = float(np.std(y_true))                       # ddof = 0 (population std of the test targets)
    var = std ** 2
    nrmse = rmse / std if std > 0 else float("nan")
    r2 = 1.0 - mse / var if var > 0 else float("nan")
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, nrmse=nrmse, finite_pred=finite)


def inverse_target(y_scaled, yscaler) -> np.ndarray:
    """Undo the S1 target standardisation: y = y_scaled * std + mean (float64)."""
    y = np.asarray(y_scaled, dtype=np.float64).ravel()
    return y * float(yscaler.std) + float(yscaler.mean)


def predict_original(model, X, yscaler, device: str = "cpu"):
    """Model prediction on X, returned in original units, plus inference time in ms."""
    import torch
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_scaled = model(X)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
    return inverse_target(y_scaled.detach().cpu().double().numpy(), yscaler), ms


def test_metrics(model, data: dict, device: str = "cpu", return_pred: bool = False):
    """The one function every member calls after training.

    Ground truth comes from the RAW test file (data["raw"]["test"][:, -1]) when S1 provides it,
    so no rounding from the scaled float32 tensor enters y_true.
    """
    y_pred, ms = predict_original(model, data["Xte"], data["yscaler"], device)
    if "raw" in data:
        y_true = np.asarray(data["raw"]["test"][:, -1], dtype=np.float64)
    else:
        y_true = inverse_target(data["yte"].cpu().double().numpy(), data["yscaler"])
    out = dict(infer_time_ms=ms, **regression_metrics(y_true, y_pred))
    return (out, y_pred) if return_pred else out


def save_predictions(folder: str, model_name: str, spec: str, equation: str, seed: int, y_pred) -> str:
    """Save test predictions (original units) so the final notebook can recompute every metric
    with the same code for all models: <folder>/<model>/<spec>/<equation>_s<seed>.npy"""
    path = os.path.join(folder, model_name, spec.replace("/", "_"), f"{equation}_s{seed}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(y_pred, dtype=np.float64))
    return path


def per_equation(df, by=("model", "spec")):
    """All five metrics per (model, spec, equation): median over the seeds."""
    by = list(by)
    cols = [c for c in (*METRICS, "n_params", "train_time_s") if c in df]
    out = df.groupby(by + ["equation"])[cols].median()
    if "diverged" in df:
        out["diverged_runs"] = df.groupby(by + ["equation"])["diverged"].sum()
    return out.reset_index()


def summarize(df, by=("model", "spec")):
    """Cross-equation summary: median and IQR of NRMSE (headline) and R^2, computed from the
    seed medians of each equation. MSE / RMSE / MAE are not aggregated (different units per
    equation); see per_equation(). Diverged runs are kept (NRMSE = inf counts as worst)."""
    by = list(by)
    pe = per_equation(df, by)
    g = pe.groupby(by)
    q = lambda p: (lambda s: s.quantile(p))
    out = g.agg(equations=("nrmse", "size"),
                nrmse_median=("nrmse", "median"), nrmse_q1=("nrmse", q(.25)), nrmse_q3=("nrmse", q(.75)),
                r2_median=("r2", "median"), r2_q1=("r2", q(.25)), r2_q3=("r2", q(.75)))
    if "diverged_runs" in pe:
        out["diverged_runs"] = g["diverged_runs"].sum()
    for c, name in (("n_params", "params_median"), ("train_time_s", "train_time_median_s")):
        if c in pe:
            out[name] = g[c].median()
    return out.reset_index()

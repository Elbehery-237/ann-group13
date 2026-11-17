"""Smoke-test the shared production data loader and training harness.

Run from the repository root after the data owner supplies ``src.data.load_problem``:

    SRSD_ROOT=/path/to/srsd-feynman_medium python scripts/phase_c_data_smoke.py

The script uses one known linear Medium equation, keeps the official test split out
of training/validation, and reports test metrics only after the shared trainer returns.
It intentionally fails with a clear message while ``src/data.py`` is still a stub.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data import load_problem
except ImportError as exc:  # expected until the owner implements the production loader
    raise SystemExit(
        "Production smoke test unavailable: src.data.load_problem is not implemented. "
        "Wait for the S1 owner to provide it; do not substitute tests.conftest.load_srsd."
    ) from exc

from src.train import set_seed, train


EQUATION = "feynman-i.29.4"  # Medium linear law used by the existing trainer acceptance test.
SEED = 13


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    root = os.environ.get("SRSD_ROOT")
    require(bool(root), "Set SRSD_ROOT to the SRSD-Feynman Medium dataset directory.")
    root_path = Path(root).expanduser().resolve()
    require(root_path.is_dir(), f"SRSD_ROOT does not exist: {root_path}")
    require(root_path.name == "srsd-feynman_medium", f"Expected Medium dataset root, got: {root_path}")

    # The intended project call is documented in the Youssef Plan's runner example.
    data = load_problem(EQUATION)
    required = ("Xtr", "ytr", "Xva", "yva", "Xte", "yte", "xscaler", "yscaler", "raw")
    missing = [key for key in required if key not in data]
    require(not missing, f"load_problem({EQUATION!r}) is missing required keys: {missing}")

    Xtr, ytr, Xva, yva, Xte, yte = (data[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))
    for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        require(isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor), f"{split}: expected tensors")
        require(X.ndim == 2 and y.ndim == 2 and y.shape[1] == 1,
                f"{split}: expected X (n,d), y (n,1); got {tuple(X.shape)}, {tuple(y.shape)}")
        require(X.dtype == torch.float32 and y.dtype == torch.float32,
                f"{split}: expected float32 tensors after safe preprocessing")
        require(X.shape[0] == y.shape[0] and X.shape[0] > 0, f"{split}: row counts do not match")
        require(torch.isfinite(X).all().item() and torch.isfinite(y).all().item(),
                f"{split}: non-finite preprocessed values")

    d = Xtr.shape[1]
    require(1 <= d <= 6, f"Expected 1–6 inputs, got {d}")
    require(all(data[k].shape[1] == d for k in ("Xva", "Xte")), "Input dimensions differ across splits")
    require(int(data.get("d", d)) == d, f"Loader metadata d={data.get('d')} does not match tensor width {d}")
    xs, ys = data["xscaler"], data["yscaler"]
    require(all(hasattr(xs, attr) for attr in ("lo", "hi", "log_cols")),
            "xscaler must expose lo, hi, and log_cols for symbolic inversion")
    require(all(hasattr(ys, attr) for attr in ("mean", "std")),
            "yscaler must expose mean and std")
    require(np.isfinite(float(ys.mean)) and np.isfinite(float(ys.std)) and float(ys.std) > 0,
            "Invalid train-fitted target scaler")

    # Verify the scaler metadata was fitted from the training split, not validation/test.
    require(all(split in data["raw"] for split in ("train", "val", "test")),
            "raw must retain all official splits for integrity checks and original-unit evaluation")
    raw_train = np.asarray(data["raw"]["train"], dtype=np.float64)
    raw_val = np.asarray(data["raw"]["val"], dtype=np.float64)
    raw_test = np.asarray(data["raw"]["test"], dtype=np.float64)
    for split, raw, tensor_x in (("train", raw_train, Xtr), ("val", raw_val, Xva), ("test", raw_test, Xte)):
        expected_rows = {"train": 8000, "val": 1000, "test": 1000}[split]
        require(raw.shape == (expected_rows, d + 1),
                f"Raw {split} split has unexpected shape {raw.shape}")
        require(raw.shape[0] == tensor_x.shape[0], f"Raw and tensor {split} row counts differ")
        require(np.isfinite(raw).all(), f"Raw {split} split contains non-finite values")

    target_train = raw_train[:, -1]
    train_mean = float(target_train.mean(dtype=np.float64))
    train_std = float(target_train.std(dtype=np.float64))
    require(np.isclose(float(ys.mean), train_mean, rtol=1e-10, atol=0.0),
            "Target mean does not match the raw training split")
    require(np.isclose(float(ys.std), train_std, rtol=1e-10, atol=0.0),
            "Target std does not match the raw training split")

    log_cols = set(xs.log_cols)
    require(all(0 <= i < d for i in log_cols), "xscaler.log_cols contains an invalid input index")
    lo, hi = np.asarray(xs.lo, dtype=np.float64), np.asarray(xs.hi, dtype=np.float64)
    require(lo.shape == (d,) and hi.shape == (d,), "Input scaler bounds must have one entry per input")
    require(np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo),
            "Input scaler bounds must be finite and have positive ranges")
    transformed_train_x = raw_train[:, :d].copy()
    for i in log_cols:
        require(np.all(transformed_train_x[:, i] > 0), f"Log-scaled input column {i} is not positive")
        transformed_train_x[:, i] = np.log10(transformed_train_x[:, i])
    require(np.allclose(lo, transformed_train_x.min(axis=0), rtol=1e-10, atol=0.0),
            "Input scaler lower bounds do not match the training split")
    require(np.allclose(hi, transformed_train_x.max(axis=0), rtol=1e-10, atol=0.0),
            "Input scaler upper bounds do not match the training split")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    model = nn.Linear(d, 1)
    model, history, info = train(
        model,
        data,
        lr=1e-2,
        max_epochs=300,
        patience=50,
        device=device,
    )
    require(not info["diverged"], f"Shared trainer marked the run diverged: {info}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte.to(device)).detach().cpu().numpy().astype(np.float64).reshape(-1)
    std, mean = float(ys.std), float(ys.mean)
    pred_original = pred_scaled * std + mean
    target_original = raw_test[:, -1]
    residual = pred_original - target_original
    test_mse = float(np.mean(residual * residual, dtype=np.float64))
    test_mae = float(np.mean(np.abs(residual), dtype=np.float64))
    require(np.isfinite(pred_original).all(), "Original-unit test predictions are non-finite")
    require(np.isfinite(test_mse) and np.isfinite(test_mae), "Test metrics are non-finite")

    report = {
        "equation": EQUATION,
        "dataset_root": str(root_path),
        "seed": SEED,
        "device": device,
        "input_dim": d,
        "shapes": {k: list(data[k].shape) for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte")},
        "all_processed_splits_finite": True,
        "train_info": {
            "epochs": int(info["epochs"]),
            "best_epoch": int(info["best_epoch"]),
            "best_val_mse_scaled": float(info["best_val"]),
            "train_time_seconds": float(info["time"]),
            "diverged": bool(info["diverged"]),
        },
        "history_epochs": {key: len(values) for key, values in history.items()},
        "test_mse_original_units": test_mse,
        "test_mae_original_units": test_mae,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

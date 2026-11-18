"""SRSD loader for the B-spline KAN experiments. Owner: Abdelrahman Elbehery.

A self-contained stand-in that follows the team's S1 data contract, so this project runs on
its own. If the team's src/data.py is used instead, it must return the same keys.

Contract returned by load_problem(name):
    d, name                       number of inputs, problem name
    Xtr, Xva, Xte                 float32 tensors (n, d), inputs min-max scaled to [-1, 1]
    ytr, yva, yte                 float32 tensors (n, 1), targets standardised
    xscaler: lo, hi, log_cols     train-split min/max of each (optionally log10) input column
    yscaler: mean, std            train-split mean/std of the target
    raw: {"train","val","test"}   the original float64 arrays (last column = target)

All statistics come from the TRAIN split only (no leakage). Scaling is done in float64 and
only then converted to float32: SRSD Medium targets span ~1e-37 … ~2e38, which would
underflow/overflow in float32. Metrics are always computed after undoing the target
scaling (see src/bkan_eval.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SPLITS = (("train", "tr"), ("val", "va"), ("test", "te"))


def srsd_root() -> Path:
    default = Path(__file__).resolve().parents[1] / "data" / "srsd-feynman_medium"
    return Path(os.environ.get("SRSD_ROOT", default))


def list_problems(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root else srsd_root()
    return sorted(p.stem for p in (root / "true_eq").glob("*.pkl"))


def wide_positive_cols(X: np.ndarray, ratio: float = 100.0) -> list[int]:
    """Columns that are always positive and span more than `ratio` (max/min) on train."""
    lo, hi = X.min(0), X.max(0)
    return [i for i in range(X.shape[1]) if lo[i] > 0 and hi[i] / lo[i] > ratio]


def load_problem(name: str, root: str | Path | None = None, log_wide: bool = False) -> dict:
    """Load one SRSD problem. log_wide=True applies log10 to always-positive wide columns
    before min-max scaling (off by default; formulas then contain log10(x))."""
    root = Path(root) if root else srsd_root()
    raw = {s: np.loadtxt(root / s / f"{name}.txt", dtype=np.float64, ndmin=2) for s, _ in SPLITS}
    d = raw["train"].shape[1] - 1
    log_cols = set(wide_positive_cols(raw["train"][:, :d])) if log_wide else set()

    def g(X):
        X = X.copy()
        for i in log_cols:
            X[:, i] = np.log10(X[:, i])
        return X

    gtr = g(raw["train"][:, :d])
    xs = SimpleNamespace(lo=gtr.min(0), hi=gtr.max(0), log_cols=log_cols)
    if np.any(xs.hi == xs.lo):
        raise ValueError(f"{name}: an input column is constant on the train split")
    ytr = raw["train"][:, d]
    ys = SimpleNamespace(mean=float(ytr.mean()), std=float(ytr.std()))
    if not ys.std > 0:
        raise ValueError(f"{name}: target is constant on the train split")

    out = dict(d=d, name=name, xscaler=xs, yscaler=ys, raw=raw)
    for s, k in SPLITS:
        Xs = 2 * (g(raw[s][:, :d]) - xs.lo) / (xs.hi - xs.lo) - 1        # float64
        ysc = (raw[s][:, d] - ys.mean) / ys.std                           # float64
        out[f"X{k}"] = torch.tensor(Xs, dtype=torch.float32)
        out[f"y{k}"] = torch.tensor(ysc, dtype=torch.float32).unsqueeze(1)
    return out

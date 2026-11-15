"""Test fixtures. `load_srsd` is a minimal stand-in for S1 (src/data.py, Jomana) used only by
tests: it follows the agreed contract (inputs min-max to [-1, 1] with optional log10 columns,
target standardized, statistics from the train split only)."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRSD_ROOT = Path(os.environ.get("SRSD_ROOT", ROOT / "data" / "srsd-feynman_medium"))
PROBLEMS = sorted(p.stem for p in (SRSD_ROOT / "true_eq").glob("*.pkl")) if SRSD_ROOT.exists() else []

needs_data = pytest.mark.skipif(not PROBLEMS, reason=f"SRSD data not found at {SRSD_ROOT} (set SRSD_ROOT)")


def first_present(*names):
    """First of `names` that exists in the selected SRSD set (tests work on Easy or Medium)."""
    return next((n for n in names if n in PROBLEMS), None)


def load_srsd(name, log_cols=()):
    raw = {s: np.loadtxt(SRSD_ROOT / s / f"{name}.txt") for s in ("train", "val", "test")}
    d = raw["train"].shape[1] - 1
    log_cols = set(log_cols)

    def g(X):
        X = X.copy()
        for i in log_cols:
            X[:, i] = np.log10(X[:, i])
        return X

    gtr = g(raw["train"][:, :d])
    xs = SimpleNamespace(lo=gtr.min(0), hi=gtr.max(0), log_cols=log_cols)
    ytr = raw["train"][:, d]
    ys = SimpleNamespace(mean=float(ytr.mean()), std=float(ytr.std()))

    out = dict(d=d, name=name, xscaler=xs, yscaler=ys, raw=raw)
    for s, k in (("train", "tr"), ("val", "va"), ("test", "te")):
        Xs = 2 * (g(raw[s][:, :d]) - xs.lo) / (xs.hi - xs.lo) - 1
        ysc = (raw[s][:, d] - ys.mean) / ys.std
        out[f"X{k}"] = torch.tensor(Xs, dtype=torch.float32)
        out[f"y{k}"] = torch.tensor(ysc, dtype=torch.float32).unsqueeze(1)
    return out

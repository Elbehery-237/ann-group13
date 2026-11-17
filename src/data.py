"""S1 — Production SRSD data loader and train-fitted preprocessing.
Owner: Jomana Mohamed.

The public contract is ``load_problem(eq)``. Set ``SRSD_ROOT`` to the root of an
SRSD-Feynman dataset (the default is this repository's data/srsd-feynman_medium).
All statistics are fitted on the official train split and then reused unchanged.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "srsd-feynman_medium"
_EXPECTED_ROWS = {"train": 8000, "val": 1000, "test": 1000}
_EQUATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class InputScaler:
    """Train-fitted bounds after any requested log10 transformations."""
    lo: np.ndarray
    hi: np.ndarray
    log_cols: frozenset[int]


@dataclass(frozen=True)
class TargetScaler:
    """Train-fitted target standardization metadata in original units."""
    mean: float
    std: float


def _load_split(path: Path, split: str, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} data file: {path}")
    try:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not parse {split} data file {path}: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != expected_rows or not 2 <= arr.shape[1] <= 7:
        raise ValueError(
            f"Malformed {split} data in {path}: expected ({expected_rows}, d+1) "
            f"with 1<=d<=6; found {arr.shape}"
        )
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))[0]
        raise ValueError(f"Non-finite value in {path} at row {bad[0]}, column {bad[1]}")
    return arr


def load_problem(eq: str, *, root: str | Path | None = None,
                 log_cols: Iterable[int] = ()) -> dict:
    """Load an equation's official SRSD splits and transform using train statistics.

    Parameters
    ----------
    eq: equation stem, e.g. ``"feynman-i.29.4"`` (without ``.txt``).
    root: optional dataset root; otherwise ``SRSD_ROOT`` or the repository default.
    log_cols: optional input-column indices to transform with log10 before min-max
        scaling. Every selected value in every split must be positive.

    Returns tensors ``Xtr/ytr``, ``Xva/yva``, ``Xte/yte`` for the shared trainer,
    plus ``d``, ``name``, the unchanged raw split arrays, and scaler objects exposing
    ``lo``, ``hi``, ``log_cols``, ``mean`` and ``std`` for symbolic inversion.
    """
    if not isinstance(eq, str) or not _EQUATION_NAME.fullmatch(eq):
        raise ValueError(f"Invalid equation name {eq!r}; pass an equation stem, not a path")
    dataset_root = Path(root or os.environ.get("SRSD_ROOT", DEFAULT_ROOT)).expanduser()
    split_dirs = {split: dataset_root / split for split in _EXPECTED_ROWS}
    missing_dirs = [str(p) for p in split_dirs.values() if not p.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"SRSD dataset is incomplete at {dataset_root}; missing split directories: "
            + ", ".join(missing_dirs)
        )

    raw = {split: _load_split(split_dirs[split] / f"{eq}.txt", split, rows)
           for split, rows in _EXPECTED_ROWS.items()}
    d = raw["train"].shape[1] - 1
    for split, arr in raw.items():
        if arr.shape[1] != d + 1:
            raise ValueError(f"{split} {eq}: expected {d} input columns and one target; got {arr.shape[1]-1}")

    try:
        requested_logs = tuple(log_cols)
    except TypeError as exc:
        raise ValueError("log_cols must be an iterable of integer input-column indices") from exc
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in requested_logs):
        raise ValueError(f"log_cols must contain integer input-column indices; got {requested_logs!r}")
    logs = frozenset(int(i) for i in requested_logs)
    if any(i < 0 or i >= d for i in logs):
        raise ValueError(f"log_cols must contain input indices in [0, {d}); got {sorted(logs)}")

    def transform_inputs(X: np.ndarray, split: str) -> np.ndarray:
        Xg = np.asarray(X, dtype=np.float64).copy()
        for i in logs:
            if np.any(Xg[:, i] <= 0):
                raise ValueError(f"{split} {eq}: log10 input column {i} contains non-positive values")
            Xg[:, i] = np.log10(Xg[:, i])
        if not np.isfinite(Xg).all():
            raise ValueError(f"{split} {eq}: input preprocessing produced non-finite values")
        return Xg

    train_X = transform_inputs(raw["train"][:, :d], "train")
    lo, hi = train_X.min(axis=0), train_X.max(axis=0)
    constant = np.flatnonzero(hi <= lo)
    if constant.size:
        raise ValueError(f"{eq}: input columns constant in training data: {constant.tolist()}")
    train_y = raw["train"][:, d]
    y_mean = float(train_y.mean(dtype=np.float64))
    y_std = float(train_y.std(dtype=np.float64))
    if not np.isfinite(y_mean) or not np.isfinite(y_std) or y_std <= 0:
        raise ValueError(f"{eq}: target training mean/std are invalid (mean={y_mean}, std={y_std})")

    xs, ys = InputScaler(lo=lo.copy(), hi=hi.copy(), log_cols=logs), TargetScaler(y_mean, y_std)
    result = {"d": d, "name": eq, "xscaler": xs, "yscaler": ys, "raw": raw}
    for split, short in (("train", "tr"), ("val", "va"), ("test", "te")):
        Xg = train_X if split == "train" else transform_inputs(raw[split][:, :d], split)
        Xscaled = 2.0 * (Xg - xs.lo) / (xs.hi - xs.lo) - 1.0
        yscaled = (raw[split][:, d] - ys.mean) / ys.std
        if not np.isfinite(Xscaled).all() or not np.isfinite(yscaled).all():
            raise ValueError(f"{split} {eq}: scaling produced non-finite values")
        Xt = torch.as_tensor(Xscaled, dtype=torch.float32)
        yt = torch.as_tensor(yscaled[:, None], dtype=torch.float32)
        if not torch.isfinite(Xt).all().item() or not torch.isfinite(yt).all().item():
            raise ValueError(f"{split} {eq}: values overflowed float32 tensor conversion")
        result[f"X{short}"] = Xt
        result[f"y{short}"] = yt
    return result

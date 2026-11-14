"""S2 — Shared training harness. Owner: Abdelrahman Elbehery.

Every accuracy result in the paper goes through `train()`, so all four models get the
same optimizer (Adam), loss (MSE on scaled targets), epoch budget and early-stopping rule.

Other members import this module; do not copy or edit it. Any change affects all four
models, so announce it to the team first.

Usage
-----
    from src.train import set_seed, train
    set_seed(0)
    model, history, info = train(model, data, hooks=(...))

`data` follows the S1 contract: tensors "Xtr", "ytr", "Xva", "yva" (X: (n, d) scaled,
y: (n, 1) scaled). The test split is never touched here.
"""
from __future__ import annotations

import copy
import math
import random
import time
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Team-wide defaults — the single place the training budget is defined.
# Agree on these once; change only with an announcement.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    lr=1e-3,
    max_epochs=5000,
    patience=300,
    batch_size=None,         # None -> full batch (SRSD train split = 8000 rows)
    full_batch_max=10_000,   # if batch_size is None and n_train exceeds this, use mini-batches
    auto_batch_size=1024,
)

# hook(model, epoch, Xtr) — runs after each epoch's parameter updates (e.g. KAN grid update)
Hook = Callable[[nn.Module, int, torch.Tensor], None]

REQUIRED_KEYS = ("Xtr", "ytr", "Xva", "yva")


def set_seed(s: int) -> None:
    """Seed every RNG the project uses (Python, NumPy, PyTorch CPU and CUDA)."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def batches(X: torch.Tensor, y: torch.Tensor, batch_size: int | None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (xb, yb): the full batch if batch_size is None, else shuffled mini-batches."""
    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        yield X, y
        return
    perm = torch.randperm(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]


def _check_data(data: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"data is missing {missing}; expected keys {REQUIRED_KEYS}")
    for split in ("tr", "va"):
        X, y = data[f"X{split}"], data[f"y{split}"]
        if X.ndim != 2 or y.ndim != 2 or y.shape[1] != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(f"bad shapes for split '{split}': X {tuple(X.shape)}, y {tuple(y.shape)}; "
                             "expected X (n, d) and y (n, 1)")


def train(
    model: nn.Module,
    data: dict,
    lr: float = DEFAULTS["lr"],
    max_epochs: int = DEFAULTS["max_epochs"],
    patience: int = DEFAULTS["patience"],
    batch_size: int | None = DEFAULTS["batch_size"],
    hooks: Sequence[Hook] = (),
    device: str = "cpu",
) -> tuple[nn.Module, dict, dict]:
    """Train `model` with Adam and early stopping on validation MSE.

    Returns
    -------
    model   : with the best-validation weights restored
    history : {"train": [train MSE per epoch], "val": [val MSE per epoch]}
    info    : {"diverged": bool, "epochs": epochs run, "best_epoch": 0-based index,
               "best_val": best val MSE, "time": wall-clock seconds}

    Divergence = a non-finite validation loss. Training stops at once, the best weights
    seen so far are restored, and info["diverged"] is True (report it, don't drop the run).
    """
    _check_data(data)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, ytr, Xva, yva = (data[k].to(device) for k in REQUIRED_KEYS)

    if batch_size is None and Xtr.shape[0] > DEFAULTS["full_batch_max"]:
        batch_size = DEFAULTS["auto_batch_size"]

    best, best_ep, best_state, wait = float("inf"), -1, None, 0
    hist: dict[str, list[float]] = {"train": [], "val": []}
    diverged = False
    t0 = time.perf_counter()
    ep = -1
    for ep in range(max_epochs):
        model.train()
        for xb, yb in batches(Xtr, ytr, batch_size):
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        for h in hooks:
            h(model, ep, Xtr)

        model.eval()
        with torch.no_grad():
            tr = F.mse_loss(model(Xtr), ytr).item()
            va = F.mse_loss(model(Xva), yva).item()
        hist["train"].append(tr)
        hist["val"].append(va)

        if not math.isfinite(va):
            diverged = True
            break
        if va < best:
            best, best_ep, best_state, wait = va, ep, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, dict(diverged=diverged, epochs=ep + 1, best_epoch=best_ep,
                             best_val=best, time=time.perf_counter() - t0)

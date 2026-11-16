"""B-spline KAN (KAN-1 … KAN-4).

Owner: Abdelrahman Elbehery.

Each edge (i -> j) carries a learnable 1-D function
    phi(x) = w_b * silu(x) + w_s * sum_k c_k B_k(x)
with B_k order-k B-splines on G intervals over [-1, 1]; each node sums its inputs
(MultKAN: some nodes multiply). Built on pykan (version pinned in requirements.txt).
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
from torch import nn

from kan import KAN


class BSplineKAN(nn.Module):
    """Wrapper so pykan follows the team model contract.

    hidden: tuple of layer specs. An int n -> n sum nodes. A pair [n_sum, n_mult]
            -> MultKAN layer with multiplication nodes (KAN-4: hidden=([5, 2],)).
    """

    def __init__(self, in_dim: int, hidden: Sequence = (5,), grid: int = 5, k: int = 3,
                 seed: int = 0, grid_eps: float = 0.02, mult_arity: int = 2, device: str = "cpu",
                 ckpt_path: str = ".pykan"):
        super().__init__()
        # pykan 0.2.8 appends to <ckpt_path>/history.txt on prune()/auto_symbolic()
        # even with auto_save=False, so the folder must exist (git-ignored).
        os.makedirs(ckpt_path, exist_ok=True)
        width = [in_dim, *[list(h) if isinstance(h, (list, tuple)) else h for h in hidden], 1]
        self.kan = KAN(width=width, grid=grid, k=k, seed=seed, grid_range=[-1, 1],
                       grid_eps=grid_eps, mult_arity=mult_arity, auto_save=False,
                       ckpt_path=ckpt_path, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def formula(self) -> "sympy.Expr":
        """Symbolic formula in scaled inputs x_1..x_d (pykan naming; see src.symbolic.to_original).

        Passes the variables explicitly: pykan 0.2.8's default path creates them with exec(),
        which fails on Python >= 3.13 (NameError: name 'x1' is not defined).
        """
        import sympy
        d = self.kan.width_in[0]
        return self.kan.symbolic_formula(var=[sympy.Symbol(f"x_{i}") for i in range(1, d + 1)])[0][0]

    def count_params(self, trainable_only: bool = True) -> int:
        """Report this, not the closed-form estimate: pykan also stores symbolic-branch parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def make_grid_update_hook(epochs: Sequence[int] = (0, 50, 100)):
    """Hook for src.train.train(hooks=...): refit the spline grid to the training samples
    at a few early epochs (suggested values; confirm on validation)."""
    epochs = set(epochs)

    def hook(model: BSplineKAN, ep: int, Xtr: torch.Tensor) -> None:
        if ep in epochs:
            with torch.no_grad():
                model.kan.update_grid(Xtr)   # pykan 0.2.8; older: update_grid_from_samples
    return hook


grid_update_hook = make_grid_update_hook()


def train_grid_extension(model: BSplineKAN, data: dict, grids: Sequence[int] = (5, 10, 20),
                         train_kw: dict | None = None, hooks=()):
    """KAN-2: train at grids[0], refine to each next grid and continue training.

    refine() returns a new pykan model with new parameters, so each stage calls
    train() again (fresh Adam state). Returns (model, stages) where stages is a list of
    {"grid", "history", "info"} — the per-stage validation errors give the staircase curve.
    """
    from src.train import train

    train_kw = dict(train_kw or {})
    stages = []
    for s, G in enumerate(grids):
        if s > 0:
            model.kan = model.kan.refine(G)
        model, hist, info = train(model, data, hooks=hooks if s == 0 else (), **train_kw)
        stages.append(dict(grid=G, history=hist, info=info))
        if info["diverged"]:
            break
    return model, stages

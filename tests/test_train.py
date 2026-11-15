"""Acceptance tests for S2 (src/train.py)."""
import math

import pytest
import torch
from torch import nn

from conftest import first_present, load_srsd, needs_data
from src.train import set_seed, train


def _split(X, y, n_tr):
    return dict(Xtr=X[:n_tr], ytr=y[:n_tr], Xva=X[n_tr:], yva=y[n_tr:])


def _mlp(d, h=64):
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))


# --- plan "done when" conditions -------------------------------------------------------
def test_linear_model_fits_2x_plus_1_exactly():
    set_seed(0)
    X = torch.linspace(-1, 1, 200).unsqueeze(1)
    X = X[torch.randperm(200)]
    model, hist, info = train(nn.Linear(1, 1), _split(X, 2 * X + 1, 150), lr=0.1, max_epochs=3000)
    assert not info["diverged"]
    assert model.weight.item() == pytest.approx(2, abs=1e-4)
    assert model.bias.item() == pytest.approx(1, abs=1e-4)


def test_early_stopping_triggers_on_overfitting_mlp():
    set_seed(0)
    X = torch.rand(40, 1) * 2 - 1
    y = torch.sin(3 * X) + 0.5 * torch.randn_like(X)          # 20 noisy training points
    patience = 100
    model, hist, info = train(_mlp(1, 256), _split(X, y, 20), lr=1e-2, max_epochs=5000, patience=patience)
    assert info["epochs"] < 5000
    assert info["epochs"] == info["best_epoch"] + 1 + patience
    assert hist["train"][-1] < hist["train"][info["best_epoch"]]   # train kept improving...
    assert min(hist["val"]) == info["best_val"]                     # ...validation did not


# --- behaviour --------------------------------------------------------------------------
def test_best_weights_are_restored():
    set_seed(0)
    X = torch.rand(40, 1) * 2 - 1
    y = torch.sin(3 * X) + 0.5 * torch.randn_like(X)
    data = _split(X, y, 20)
    model, hist, info = train(_mlp(1, 256), data, lr=1e-2, patience=50)
    with torch.no_grad():
        val = nn.functional.mse_loss(model(data["Xva"]), data["yva"]).item()
    assert val == pytest.approx(info["best_val"], rel=1e-5)


def test_divergence_is_flagged_and_weights_stay_finite():
    X = torch.linspace(-1, 1, 50).unsqueeze(1)
    model, hist, info = train(nn.Linear(1, 1), _split(X, 1e30 * X, 40), lr=1e10, max_epochs=50)
    assert info["diverged"]
    assert all(torch.isfinite(p).all() for p in model.parameters()) or info["best_epoch"] == -1


def test_same_seed_same_history():
    def run():
        set_seed(123)
        X = torch.rand(100, 2) * 2 - 1
        return train(_mlp(2), _split(X, X[:, :1] * X[:, 1:], 80), max_epochs=50)[1]
    assert run() == run()


def test_minibatch_path():
    set_seed(0)
    X = torch.rand(500, 2) * 2 - 1
    model, hist, info = train(_mlp(2), _split(X, X.sum(1, keepdim=True), 400), lr=1e-2,
                              max_epochs=100, batch_size=64)
    assert hist["val"][-1] < hist["val"][0]


def test_hooks_called_every_epoch():
    calls = []
    X = torch.rand(30, 1)
    train(nn.Linear(1, 1), _split(X, X, 20), max_epochs=7, hooks=(lambda m, ep, Xtr: calls.append(ep),))
    assert calls == list(range(7))


def test_bad_shapes_rejected():
    X = torch.rand(30, 1)
    with pytest.raises(ValueError):
        train(nn.Linear(1, 1), dict(Xtr=X, ytr=X.squeeze(1), Xva=X, yva=X))
    with pytest.raises(KeyError):
        train(nn.Linear(1, 1), dict(Xtr=X, ytr=X))


# --- real SRSD data ---------------------------------------------------------------------
@needs_data
def test_srsd_linear_law_fit_exactly():
    # A law linear in its input (Medium: i.29.4 = 3.3e-9*x0; Easy: iii.12.43 = 3.313e-34*x0/pi).
    # Min-max and standardization are affine, so in scaled space this is a line:
    # a linear model must reach ~zero validation error.
    name = first_present("feynman-i.29.4", "feynman-iii.12.43")
    if name is None:
        pytest.skip("no linear law in this set")
    data = load_srsd(name)
    set_seed(0)
    model, hist, info = train(nn.Linear(data["d"], 1), data, lr=1e-2, max_epochs=3000)
    assert info["best_val"] < 1e-8


@needs_data
def test_srsd_product_law_mlp_learns():
    # A product law (Medium: i.39.10 = 1.5*x0*x1; Easy: i.12.1 = x0*x1).
    # Standardized target -> predicting the mean gives MSE ~ 1.
    name = first_present("feynman-i.39.10", "feynman-i.12.1")
    if name is None:
        pytest.skip("no product law in this set")
    data = load_srsd(name)
    set_seed(0)
    model, hist, info = train(_mlp(data["d"]), data, lr=1e-2, max_epochs=1500, patience=200)
    assert not info["diverged"] and math.isfinite(info["best_val"])
    assert info["best_val"] < 1e-3

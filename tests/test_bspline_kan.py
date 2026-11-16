"""B-spline KAN smoke tests (plan steps 3-4)."""
import torch

from src.models import REGISTRY
from src.models.bspline_kan import BSplineKAN, make_grid_update_hook, train_grid_extension
from src.train import set_seed, train


def _data(d=3, n=200):
    set_seed(0)
    X = torch.rand(n, d) * 2 - 1
    y = (X[:, :1] * X[:, 1:2] + torch.sin(X[:, 2:3]))
    k = int(0.8 * n)
    return dict(Xtr=X[:k], ytr=y[:k], Xva=X[k:], yva=y[k:])


def test_registry_and_contract():
    m = REGISTRY["bkan"](in_dim=3, hidden=(5,), grid=5, k=3)
    assert m(torch.zeros(7, 3)).shape == (7, 1)
    assert m.count_params() > 0


def test_trains_through_harness_with_grid_hook():
    data = _data()
    m = BSplineKAN(3, hidden=(5,), grid=5, k=3, seed=0)
    m, hist, info = train(m, data, lr=1e-2, max_epochs=150, patience=50,
                          hooks=(make_grid_update_hook((0, 50)),))
    assert not info["diverged"]
    assert min(hist["val"]) < hist["val"][0]


def test_multkan_kan4():
    m = BSplineKAN(3, hidden=([5, 2],), grid=5, k=3)
    assert m(torch.zeros(4, 3)).shape == (4, 1)


def test_grid_extension_stages():
    m = BSplineKAN(3, hidden=(5,), grid=5, k=3, seed=0)
    m, stages = train_grid_extension(m, _data(), grids=(5, 10),
                                     train_kw=dict(lr=1e-2, max_epochs=60, patience=30))
    assert [s["grid"] for s in stages] == [5, 10]

"""Acceptance tests for S5 (src/symbolic.py)."""
import time

import numpy as np
import pytest
import sympy as sp
from types import SimpleNamespace

from conftest import PROBLEMS, SRSD_ROOT, first_present, load_srsd, needs_data
from src.data import load_problem
from src.symbolic import (is_exact, load_true_eq, ned, predict, round_floats,
                          to_original, with_timeout)

x0, x1, x2 = sp.symbols("x0 x1 x2")


def _sleep(s):
    time.sleep(s)
    return "done"


def scaled_form(true, xs, ys, d):
    """Write the ground truth in scaled variables x_1..x_d and scaled target — what a perfect
    model working on S1-scaled data would return."""
    u = sp.symbols(f"x_1:{d + 1}")
    subs = {}
    def decimal(value):
        return sp.Rational(str(value))

    for i in range(d):
        lo, hi = decimal(xs.lo[i]), decimal(xs.hi[i])
        g = (u[i] + 1) / 2 * (hi - lo) + lo
        subs[sp.Symbol(f"x{i}")] = 10 ** g if i in xs.log_cols else g
    return (true.subs(subs, simultaneous=True) - decimal(ys.mean)) / decimal(ys.std)


def scaled_form_float(true, xs, ys, d):
    """Reproduce the historical forward helper's Python-float scaler arithmetic."""
    u = sp.symbols(f"x_1:{d + 1}")
    subs = {}
    for i in range(d):
        lo, hi = float(xs.lo[i]), float(xs.hi[i])
        g = (u[i] + 1) / 2 * (hi - lo) + lo
        subs[sp.Symbol(f"x{i}")] = 10 ** g if i in xs.log_cols else g
    return (true.subs(subs, simultaneous=True) - float(ys.mean)) / float(ys.std)


# --- plan "done when" conditions on every problem in the set --------------------------------
@needs_data
@pytest.mark.parametrize("name", PROBLEMS)
def test_ground_truth_is_exact_with_zero_ned(name):
    true = load_true_eq(name, SRSD_ROOT)
    assert is_exact(true, true)
    assert ned(true, true) == 0.0


@needs_data
@pytest.mark.parametrize("name", PROBLEMS)
def test_wrong_formula_is_not_exact(name):
    true = load_true_eq(name, SRSD_ROOT)
    wrong = true + sp.Symbol("x0") ** 3        # changes the structure in every problem
    assert not is_exact(wrong, true)
    assert ned(wrong, true) > 0


# --- to_original on real data -------------------------------------------------------------
@pytest.mark.parametrize("name", ["feynman-i.39.10", "feynman-ii.34.29a"])
@needs_data
def test_decimal_scaler_round_trip_regressions(name):
    """These equations exposed cancellation residuals from binary-float scaler constants."""
    data = load_problem(name, root=SRSD_ROOT)
    true = load_true_eq(name, SRSD_ROOT)
    d = data["d"]
    scaled = scaled_form(true, data["xscaler"], data["yscaler"], d)
    recovered = to_original(scaled, data["xscaler"], data["yscaler"], d)
    assert is_exact(recovered, true, round_digits=6)


@pytest.mark.parametrize("name", ["feynman-i.39.10", "feynman-ii.34.29a"])
@needs_data
def test_to_original_float_forward_regressions_against_ground_truth(name):
    """Exercise the historical float-based forward formula and compare to the pickle truth."""
    data = load_problem(name, root=SRSD_ROOT)
    true = load_true_eq(name, SRSD_ROOT)
    d = data["d"]
    scaled = scaled_form_float(true, data["xscaler"], data["yscaler"], d)
    assert scaled.atoms(sp.Float), "historical Python-float scaling must introduce SymPy Float artifacts"
    recovered = to_original(scaled, data["xscaler"], data["yscaler"], d)
    test = data["raw"]["test"]
    np.testing.assert_allclose(
        predict(recovered, test[:, :d]), test[:, d], rtol=1e-6,
        atol=1e-9 * np.abs(test[:, d]).max(),
    )
    # Float contamination is already part of the input expression. The inverse should
    # preserve predictions, but symbolic exactness is not expected on this historical path.
    assert not is_exact(recovered, true, round_digits=6)


def test_to_original_decimal_scalers_reconstruct_an_independent_linear_truth():
    xs = SimpleNamespace(lo=["0.1"], hi=["0.9"], log_cols=set())
    ys = SimpleNamespace(mean="0.3", std="0.7")
    u = sp.Symbol("x_1")
    # Independent exact-decimal forward equation for y=x0.
    scaled = (sp.Rational(1, 10) + (u + 1) * sp.Rational(2, 5) - sp.Rational(3, 10)) / sp.Rational(7, 10)
    recovered = to_original(scaled, xs, ys, d=1)
    assert sp.simplify(recovered - x0) == 0


def test_float_scaler_path_preserves_predictions_despite_float_artifacts():
    xs = SimpleNamespace(lo=[0.1], hi=[0.9], log_cols=set())
    ys = SimpleNamespace(mean=0.3, std=0.7)
    true = 2 * x0**2 + 3 * x0
    scaled = scaled_form_float(true, xs, ys, d=1)
    assert scaled.atoms(sp.Float), "Python-float scaler arithmetic must leave Float artifacts"
    recovered = to_original(scaled, xs, ys, d=1)
    probe = np.linspace(-2.0, 2.0, 101).reshape(-1, 1)
    np.testing.assert_allclose(predict(recovered, probe), predict(true, probe), rtol=1e-12, atol=1e-12)


def test_to_original_preserves_a_genuine_small_physical_constant():
    xs = SimpleNamespace(lo=["0.1"], hi=["0.9"], log_cols=set())
    ys = SimpleNamespace(mean="0.3", std="0.7")
    u = sp.Symbol("x_1")
    tiny = sp.Rational(3313, 10**37)
    true = tiny * x0 + sp.Rational(8854, 10**15)
    x_from_u = sp.Rational(1, 10) + (u + 1) * sp.Rational(2, 5)
    scaled = (true.subs(x0, x_from_u) - sp.Rational(3, 10)) / sp.Rational(7, 10)
    recovered = to_original(scaled, xs, ys, d=1)
    assert sp.simplify(recovered - true) == 0
    recovered_slope = sp.simplify(sp.diff(recovered, x0))
    assert recovered_slope != 0
    assert recovered_slope == tiny


@needs_data
@pytest.mark.parametrize("name", PROBLEMS)
def test_to_original_round_trip(name):
    data = load_srsd(name)
    true = load_true_eq(name, SRSD_ROOT)
    d = data["d"]
    back = to_original(scaled_form(true, data["xscaler"], data["yscaler"], d),
                       data["xscaler"], data["yscaler"], d)
    test = data["raw"]["test"]
    np.testing.assert_allclose(predict(back, test[:, :d]), test[:, d], rtol=1e-6,
                               atol=1e-9 * np.abs(test[:, d]).max())
    assert is_exact(back, true, round_digits=6)


@needs_data
def test_to_original_with_log_columns():
    # a problem whose inputs are all positive, so log10 scaling is allowed
    name = first_present("feynman-i.39.10", "feynman-i.12.1")
    if name is None:
        pytest.skip("no suitable problem in this set")
    data = load_srsd(name, log_cols=(0, 1))
    true = load_true_eq(name, SRSD_ROOT)
    back = to_original(scaled_form(true, data["xscaler"], data["yscaler"], 2),
                       data["xscaler"], data["yscaler"], 2)
    test = data["raw"]["test"]
    np.testing.assert_allclose(predict(back, test[:, :2]), test[:, 2], rtol=1e-6)
    assert is_exact(back, true, round_digits=6)


def test_to_original_hand_example():
    # y = x0 + log10(x1), x1 log-scaled. In scaled space: x0 = 5(u1+1), log10 x1 = u2+1,
    # y~ = (y - 3)/2.
    xs = SimpleNamespace(lo=[0.0, 0.0], hi=[10.0, 2.0], log_cols={1})
    ys = SimpleNamespace(mean=3.0, std=2.0)
    u1, u2 = sp.symbols("x_1 x_2")
    orig = to_original((5 * (u1 + 1) + (u2 + 1) - 3) / 2, xs, ys, d=2)
    assert sp.simplify(orig - (x0 + sp.log(x1, 10))) == 0


def test_to_original_rejects_constant_column():
    xs = SimpleNamespace(lo=[1.0], hi=[1.0], log_cols=set())
    with pytest.raises(ValueError):
        to_original(sp.Symbol("x_1"), xs, SimpleNamespace(mean=0, std=1), d=1)


# --- is_exact conventions -------------------------------------------------------------------
def test_scale_and_offset_count_as_exact():
    true = 3.313e-34 * x0 / sp.pi
    assert is_exact(5 * true, true)
    assert is_exact(true + 7, true)
    assert not is_exact(0 * true, true)


def test_rounding_removes_fit_noise_without_zeroing_small_constants():
    true = 8.854e-12 * x0 ** 2
    noisy = 8.8540003e-12 * x0 ** 2.0000001
    assert not is_exact(noisy, true)
    assert is_exact(noisy, true, round_digits=4)
    assert round_floats(sp.Float(3.313e-34) * x0, 4) != 0


# --- ned conventions ------------------------------------------------------------------------
def test_ned_ignores_constant_values_and_handles_missing():
    true = 8.854e-12 * x0 ** 2
    assert ned(1.0e-3 * x0 ** 2, true) == 0.0      # constants are all "Const"
    assert ned(None, true) == 1.0
    assert 0 < ned(x0 ** 2 + x0, true) <= 1


def test_ned_is_capped_at_one():
    assert ned(sp.sin(x0) * sp.exp(x1) + sp.cos(x2) * x1 ** 3, x0) == 1.0


# --- timeout --------------------------------------------------------------------------------
def test_with_timeout():
    assert with_timeout(_sleep, (0.1,), timeout=10) == "done"
    t = time.perf_counter()
    assert with_timeout(_sleep, (30,), timeout=2) is None
    assert time.perf_counter() - t < 10

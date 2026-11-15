"""S5 — Symbolic-recovery tool: score a recovered formula against the SRSD ground truth.
Owner: Abdelrahman Elbehery.

Pipeline for one equation
-------------------------
    true = load_true_eq("feynman-i.12.1")        # sympy expr in x0, x1, ... (true_eq/<name>.pkl)
    pred = to_original(expr_scaled, xs, ys, d)   # undo S1 scaling on the model's formula
    is_exact(pred, true)                         # SRBench convention: difference or ratio constant
    ned(pred, true)                              # SRSD normalized tree-edit distance, in [0, 1]

Always compare formulas in ORIGINAL variables, never in scaled ones.

Other members import this module; do not copy or edit it.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "srsd-feynman_medium"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_true_eq(name: str, root: Path | str = DEFAULT_ROOT) -> sp.Expr:
    """Load the ground-truth sympy expression for one SRSD problem.

    root: an SRSD set folder containing true_eq/ (e.g. data/srsd-feynman_medium).
    Variables are x0..x_{d-1}, in the same order as the data columns (last column = target).
    """
    path = Path(root) / "true_eq" / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(expr: sp.Expr, X: np.ndarray) -> np.ndarray:
    """Evaluate a formula in x0..x_{d-1} on rows of X (original units). Returns shape (n,)."""
    d = X.shape[1]
    x = sp.symbols(f"x0:{d}")
    f = sp.lambdify(x, expr, modules="numpy")
    out = f(*[X[:, i] for i in range(d)])
    return np.broadcast_to(np.asarray(out, dtype=float), (X.shape[0],)).copy()


# ---------------------------------------------------------------------------
# Undo S1 scaling
# ---------------------------------------------------------------------------
def to_original(expr_scaled: sp.Expr, xs: Any, ys: Any, d: int) -> sp.Expr:
    """Rewrite a formula found in scaled variables in terms of the original variables.

    S1 input scaling (column i):  x~_i = 2 (g(x_i) - lo_i) / (hi_i - lo_i) - 1
                                  g = log10 if i in xs.log_cols, else identity;
                                  lo_i, hi_i = min, max of g(x_i) on the train split.
    S1 target scaling:            y~ = (y - ys.mean) / ys.std

    So  y = ys.std * f(x~(x)) + ys.mean.

    Accepted input names in expr_scaled: x_1..x_d (pykan's default) or x0..x_{d-1}.
    Required scaler fields (interface with S1 / src/data.py):
        xs.lo, xs.hi   sequences of length d
        xs.log_cols    iterable of column indices (may be empty)
        ys.mean, ys.std
    """
    x = sp.symbols(f"x0:{d}")
    log_cols = set(getattr(xs, "log_cols", ()) or ())
    subs = {}
    for i in range(d):
        lo, hi = float(xs.lo[i]), float(xs.hi[i])
        if hi == lo:
            raise ValueError(f"column {i} is constant on the train split (lo == hi == {lo})")
        g = sp.log(x[i], 10) if i in log_cols else x[i]
        scaled = 2 * (g - lo) / (hi - lo) - 1
        subs[sp.Symbol(f"x_{i + 1}")] = scaled
        subs[sp.Symbol(f"x{i}")] = scaled
    return expr_scaled.subs(subs, simultaneous=True) * float(ys.std) + float(ys.mean)


# ---------------------------------------------------------------------------
# Timeout helper (sympy.simplify can hang). Process-based, so it also works on Windows.
# ---------------------------------------------------------------------------
def _worker(q, fn, args):
    try:
        q.put(("ok", fn(*args)))
    except Exception as e:  # noqa: BLE001
        q.put(("err", repr(e)))


def with_timeout(fn: Callable, args: tuple = (), timeout: float = 20) -> Any | None:
    """Run fn(*args) in a child process. Returns None on timeout or exception.

    fn must be a module-level (picklable) function.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, fn, args), daemon=True)
    p.start()
    try:
        status, val = q.get(timeout=timeout)   # read before join: large results can block the pipe
    except Exception:  # queue.Empty on timeout
        status, val = "timeout", None
    p.join(1)
    if p.is_alive():
        p.terminate()
        p.join()
    return val if status == "ok" else None


def _simplify(e):
    return sp.simplify(e)


# ---------------------------------------------------------------------------
# Exact recovery
# ---------------------------------------------------------------------------
def round_floats(expr: sp.Expr, digits: int) -> sp.Expr:
    """Remove fitting noise from the floats in a recovered formula.

    Each Float is rounded to `digits` SIGNIFICANT figures (decimal rounding would turn
    constants like 3.313e-34 into 0). A value that becomes a whole number is replaced by an
    exact Integer, so an exponent fitted as 2.0000001 becomes 2.
    """
    reps = {}
    for a in expr.atoms(sp.Float):
        v = float(f"{float(a):.{digits}g}")
        reps[a] = sp.Integer(int(v)) if v == int(v) and abs(v) < 1e15 else sp.Float(v, digits)
    return expr.xreplace(reps)


def _is_const(e) -> bool:
    return e is not None and len(e.free_symbols) == 0


def is_exact(pred: sp.Expr, true: sp.Expr, timeout: float = 20, round_digits: int | None = None) -> bool:
    """SRBench "symbolic solution" check: pred is exact if
        simplify(pred - true) is a constant, or simplify(pred / true) is a non-zero constant.

    Note what this allows: a correct structure with a wrong overall scale factor or offset
    still counts as exact. Report this convention in the paper.

    round_digits: if given, round the floats in BOTH pred and true to that many significant
    figures first (see round_floats), so both are compared at the same precision. Rounding
    only pred would break correct matches when the truth has long constants
    (e.g. 28235825615.541). Report the value used. If simplify exceeds `timeout` seconds, that check counts as failed.
    """
    if round_digits is not None:
        pred = round_floats(pred, round_digits)
        true = round_floats(true, round_digits)
    diff = with_timeout(_simplify, (pred - true,), timeout)
    if _is_const(diff):
        return True
    ratio = with_timeout(_simplify, (pred / true,), timeout)
    return _is_const(ratio) and ratio != 0 and ratio.is_finite is not False


# ---------------------------------------------------------------------------
# Normalized edit distance — same procedure as the SRSD benchmark
# (omron-sinicx/srsd-benchmark, MIT License: eq_comparator.load_eq_as_tree,
#  eq/conversion.sympy2zss_module, eq/eval.compute_distance).
# ---------------------------------------------------------------------------
def _srsd_canonical(expr: sp.Expr) -> sp.Expr:
    """Canonical form SRSD applies to both equations before building trees."""
    e = sp.sympify(str(expr))
    e = e.subs(sp.pi, sp.pi.evalf()).evalf().factor().simplify().subs(1.0, 1)
    return sp.sympify(str(e))


def _to_zss(expr):
    from sympy.utilities.misc import func_name
    from zss import Node

    if expr.is_number:
        label = "Const"
    elif isinstance(expr, sp.Symbol):
        label = str(expr)
    else:
        label = func_name(expr)
    node = Node(label)
    for a in expr.args:
        node.addkid(_to_zss(a))
    return node


def _count_nodes(node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _ned_worker(pred, true):
    from zss import simple_distance

    gt_tree = _to_zss(_srsd_canonical(true))
    est_tree = _to_zss(_srsd_canonical(pred))
    dist = simple_distance(est_tree, gt_tree)
    n = _count_nodes(gt_tree)
    return min(dist, n) / n


def ned(pred: sp.Expr | None, true: sp.Expr, timeout: float = 120) -> float:
    """Normalized tree-edit distance between pred and true, in [0, 1]; 0 = same tree.

    min(TED(pred, true), |true|) / |true|, where |true| is the node count of the
    ground-truth tree and every number is the same leaf "Const". As in SRSD, a missing
    prediction or one that can't be processed within `timeout` seconds scores 1.
    """
    if pred is None:
        return 1.0
    val = with_timeout(_ned_worker, (pred, true), timeout)
    return 1.0 if val is None else float(val)

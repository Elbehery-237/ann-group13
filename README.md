# ANN Group 13 — Phase 2 (26CSAI10H · Artificial Neural Networks)

## Layout

```
ann-group13/
├── data/                       # downloaded SRSD files (git-ignored)
├── configs/<model>/<spec>.yaml
├── src/
│   ├── data.py                 # S1 Jomana        load_problem(), scalers
│   ├── train.py                # S2 Elbehery      train(), set_seed()
│   ├── metrics.py, plots.py    # S3 A. Mohamed    metrics, figure style
│   ├── runner.py, aggregate.py # S4 Yousef        experiment loop, tables
│   ├── symbolic.py             # S5 Elbehery      formula check vs ground truth
│   └── models/
│       ├── __init__.py         # REGISTRY
│       ├── mlp.py, fourier_mlp.py   # Jomana
│       ├── bspline_kan.py           # Elbehery
│       ├── siren.py                 # A. Mohamed
│       └── cheby_kan.py             # Yousef
├── scripts/
├── tests/
├── results/<model>.csv         # one file per member
├── figures/
├── requirements.txt
└── README.md
```

| Member | Model | Shared piece |
|---|---|---|
| Jomana Mohamed | Fourier-feature MLP + plain MLP | S1 data pipeline, S6 baseline |
| Abdelrahman Elbehery | B-spline KAN | S2 training harness, S5 symbolic tool |
| Abdelrahman Mohamed | SIREN MLP | S3 metrics and plots |
| Yousef Basuney | Chebyshev KAN | S4 experiment runner and aggregation |

Shared modules have one owner; the others import them and do not copy or edit them.

## Dataset — SRSD Feynman (Medium)

Source: [omron-sinicx/srsd-benchmark](https://github.com/omron-sinicx/srsd-benchmark). Download the Medium set
from Hugging Face (in Colab, into the shared Drive folder):

```bash
git lfs install
git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium data/srsd-feynman_medium
```

Expected layout: `srsd-feynman_medium/{train,val,test}/<name>.txt` (space-separated, last column = target;
8000/1000/1000 rows) and `srsd-feynman_medium/true_eq/<name>.pkl` (sympy ground truth in `x0, x1, …`).
40 problems, 1–6 inputs. Constants range from about 1e-78 to 1e87, so scale targets in float64 before
converting to float32 tensors. Set `SRSD_ROOT` if the data is elsewhere (e.g. on Google Drive). The code
also works with the Easy set: point `SRSD_ROOT` at it.

## Tests

```bash
pip install -r requirements.txt
pytest -q          # data-dependent tests are skipped if the dataset is missing
```

## Evaluation metrics (same for all four models)
Computed on the **test split, in original units, after undoing the target scaling** (float64).
With e_i = ŷ_i − y_i over the n test points:

| Metric | Definition |
|---|---|
| MSE | (1/n) Σ e_i² |
| RMSE | √MSE |
| MAE | (1/n) Σ \|e_i\| |
| R² | 1 − MSE / Var(y_test) |
| NRMSE | RMSE / std(y_test), population std (ddof = 0); Petersen et al., ICLR 2021; = √(1 − R²) |

Per equation: median over seeds 0, 1, 2. MSE / RMSE / MAE keep each equation's units, so they are reported
per equation only; across equations, report the median and IQR of NRMSE (headline) and R².
Reference implementation: `regression_metrics()` in `src/bkan_eval.py`.

## Implemented so far
- `src/train.py` (S2) — `train()`, `set_seed()`, team-wide `DEFAULTS`.
- `src/symbolic.py` (S5) — `load_true_eq()`, `to_original()`, `is_exact()`, `ned()`, `predict()`.
  `ned()` reproduces the official SRSD `eq_comparator.py` procedure (MIT License).
- `src/models/bspline_kan.py` — B-spline KAN (KAN-1…KAN-4), registered as `"bkan"`.
- `src/bkan_data.py`, `src/bkan_eval.py` — loader (S1 contract) and the five metrics above, used by
  `notebooks/bkan.ipynb` so the B-spline KAN runs do not wait for `src/data.py` / `src/metrics.py`.
- `configs/bkan/*.yaml` — KAN-1, KAN-2, KAN-3, KAN-4 and symbolic-pipeline settings.
- `notebooks/bkan.ipynb` — timing pilot, accuracy runs, symbolic recovery, edge figure, result tables.

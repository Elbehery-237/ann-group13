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

# notebooks/

Colab notebooks: one shared setup notebook plus one per member. Notebooks only **call** code in `src/`;
models, training, metrics and scoring live in `src/` so all four members use the same code.

| Notebook | Owner | What it does |
|---|---|---|
| `00_setup.ipynb` | Elbehery | Mount Drive, clone/pull repo, download SRSD Medium (once), run tests, print versions |
| `bkan.ipynb` | Elbehery | B-spline KAN: timing pilot, accuracy runs, symbolic pipeline, edge figure |
| `mlp.ipynb`, `ffmlp.ipynb` | Jomana | Plain MLP and Fourier-feature MLP runs |
| `siren.ipynb` | A. Mohamed | SIREN runs |
| `ckan.ipynb` | Yousef | Chebyshev KAN runs |

Start your own notebook by copying the setup cell and the results loop from `bkan.ipynb`.

## Rules
- First cell of every notebook = the setup cell (Drive, repo, `SRSD_ROOT`, `RESULTS`, `FIGURES`, `DEVICE`).
- Write results and figures to **Drive** (`RESULTS`, `FIGURES`), never only to `/content`.
- Results loops append one row per run and skip finished runs, so a disconnect loses one run at most.
- Don't define models or metrics inside notebooks; put them in `src/` and push.
- Never paste a GitHub token into a cell: use Colab Secrets (`GITHUB_TOKEN`).
- Clear outputs before committing during development (Edit → Clear all outputs). Keep outputs only in
  the final versions if the instructor wants to see them.

## Opening and saving
- Open from GitHub: `https://colab.research.google.com/github/Elbehery-237/ann-group13/blob/master/notebooks/<name>.ipynb`
- Save back: File → Save a copy in GitHub → repo `ann-group13`, branch `master`, path `notebooks/<name>.ipynb`.

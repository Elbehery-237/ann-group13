# Phase C Verification Update

This current-status section records the latest Colab verification and supersedes conflicting status statements in the historical audit below. Phase D and overall project completion are not claimed.

## Snapshot and repository provenance

- Repository: `ann-group13`; branch: `phase-c-verification`; base commit: `8d1f6151dc6b6425034a89be0fdae361e2753f01`.
- Verified Colab source snapshot: `f301f34d454f8bb0603d70fe6ceae089ea4d6e4526486757aeb33d9b81f5f42c` (`UNCOMMITTED_WORKTREE_SNAPSHOT`).
- Immediately before this documentation-only audit update, all 45 files matched the snapshot manifest; the verified implementation, tests, and notebook are unchanged.
- Dataset root: `/content/drive/MyDrive/ann-group13/data/srsd-feynman_medium`.
- Dataset revision and checkout: `de7af9e2638494678ebe18d1174bbc6a02037eec`.
- Colab report: `/content/drive/MyDrive/ann-group13/results/phase_c_f301f34d454f_verification.json`.
- These results are from Colab; tests were not rerun locally.

## Colab verification results

| Stage | Result | Skipped |
|---|---:|---:|
| Symbolic tests | 135 passed | 0 |
| Data tests | 22 passed | 0 |
| Production smoke test | Passed on CUDA at the canonical Drive dataset path | — |
| Full test suite | 171 passed | 0 |

Phase C verification for this snapshot passed in Colab. The checkout remains uncommitted. Phase D remains deferred.

---
# Phase C Verification Status

**Branch:** `phase-c-verification`
**Verified starting point:** `ed1c0f2209d01abd20b22241ddf09b3875fd9cc2`, equal to `origin/master` when this work began. The checkout was clean before Phase C edits.

## Source instructions and current user request

### Official coursework brief

The attached `ANN_26CSCI33H_CourseworkBrief_Assignment-Project_v6_EEchanges.docx` identifies module `26CSCI33H`. It requires one common dataset across team members and says student-proposed datasets require approval. It does not select Easy, Medium, or MNIST. It also requires reproducible Colab/Jupyter code and at least four specifications per student. Its GenAI declaration requirement applies to the coursework report.

### Youssef Plan

The attached `Youssef Plan.pdf` assigns the Chebyshev KAN, shared runner/aggregation, and environment log to Yousef; `src/data.py` and its scalers to Jomana; and `src/symbolic.py` to Abdelrahman Elbehery. It says shared modules should not be edited by other members without asking the owner. Its runner example calls `load_problem(eq)` and passes the returned data to `train(model, data)`. Its full-experiment step says “all Easy equations × 3 seeds.”

### Current user direction

The current user approved SRSD-Feynman Medium for this work and asks to keep this phase on `phase-c-verification`. I treat Medium as the active task dataset. The Plan's Easy statement remains a documented source conflict, while the user's current approval resolves the dataset choice for this task. The official team Phase 1 proposal has not been provided. The earlier role prompt's module code `26CSAI10H` also differs from the coursework brief's `26CSCI33H`; confirm the course context with the group.

## Verified and pending requirements

| Requirement | Source | Status |
|---|---|---|
| Medium only for this Phase C work | Current user direction | Approved by the user. Earlier Colab output validated 40 Medium equations and the 8000/1000/1000 split sizes. |
| Team uses one dataset and proposed dataset approval is documented | Coursework brief | The coursework requirement is clear. The user now confirms Medium as approved; retain the approval/team decision in project records. |
| Production loader loads official splits and fits scalers on train only | User request, Plan, `src/train.py`, `src/symbolic.py`, and test fixture contract | Pending: production `src/data.py` is still a one-line stub. |
| Colab persists data and outputs | User request and setup notebook | The setup notebook stores data, results, checkpoints, and figures under Drive. The repository source is re-cloned under `/content` each session; do not keep uncommitted code only there. |
| Exact symbolic round-trip tests pass | Existing tests and prior Colab output | Pending owner investigation: 2 previously failing tests; no shared symbolic code or test changed. |
| Production loader smoke test | Current user request | Script prepared; cannot pass until the owner supplies `load_problem`. No result is claimed. |

## Repository status at branch head

The branch includes a new B-spline implementation and four B-spline smoke tests from the latest `master` commit. `src/models/__init__.py` registers `bkan`. Other model files, including `src/models/cheby_kan.py`, remain stubs. `src/data.py`, `src/runner.py`, `src/aggregate.py`, `src/metrics.py`, and `src/plots.py` remain stubs. `src/train.py` is the shared trainer; it expects `Xtr`, `ytr`, `Xva`, and `yva` with shapes `(n,d)` and `(n,1)`. `src/symbolic.py` expects input scaler fields `lo`, `hi`, `log_cols` and target scaler fields `mean`, `std`.

`tests/conftest.py` contains a separate test-only loader. The suite has no production data-loader tests. Existing `tests/test_train.py` has synthetic trainer coverage and two real-data cases through that fixture. The previous Colab result of 136 passed / 2 failed / 0 skipped was from the older `94256fc` tree, before the four B-spline tests were added. It is not a test result for the current `ed1c0f2` branch head. The current full suite has not been rerun in Colab.

`requirements.txt` pins `zss==1.2.0` for symbolic NED and `pykan==0.2.8` for the B-spline model. It does not pin pytest or the standard Colab scientific stack; pytest was available in the user's earlier Colab run. The current notebook reports the runtime versions and verifies imports. No dependency versions were changed in this branch work.

## Symbolic owner request — draft, not sent

**To Abdelrahman Elbehery, owner of `src/symbolic.py`:**

> Please investigate the `to_original()` round-trip failures for `feynman-i.39.10` and `feynman-ii.34.29a`. In the previous run, numerical prediction checks passed, but strict symbolic exactness found tiny residual terms. Could you check whether scaler metadata should use exact decimal SymPy values, whether forward and inverse scaling use the same representation, and whether the residuals are only numerical or reflect a mathematical mismatch? Please preserve genuine small physical constants. We should not weaken `is_exact()`, change NED behavior, or zero values by an arbitrary magnitude threshold. Please add regression coverage for both equations and a control that preserves a genuine small constant.

No changes have been made to `src/symbolic.py` or `tests/test_symbolic.py`. The earlier GitHub issue request was denied with HTTP 403; no message was sent.

## Data pipeline owner request — draft, not sent

**To Jomana Mohamed, owner of `src/data.py`:**

> Please provide the production `load_problem(eq)` loader and confirm its return contract. The Plan shows `load_problem(eq)` and scaler objects; the shared trainer requires `Xtr`, `ytr`, `Xva`, `yva`, while the existing test fixture also retains `Xte`, `yte`, raw split arrays, `xscaler`, and `yscaler`. Please confirm the production keys, then load SRSD-Feynman Medium with its official split boundaries, support 1–6 inputs, fit preprocessing only on train data, use float64-safe preprocessing before tensor conversion, and validate shapes and finite values. Please add production-loader tests rather than treating `tests/conftest.py` as production code.

No changes have been made to `src/data.py`. The earlier GitHub issue request was denied with HTTP 403; no message was sent.

## Prepared production-loader smoke test

`scripts/phase_c_data_smoke.py` uses `load_problem("feynman-i.29.4")`, a known Medium linear equation. Once the owner confirms the loader's output keys, it checks:

- Official split row counts and raw/preprocessed shapes.
- 1–6 input dimensions, float32 tensor shapes, and finite values.
- Input bounds and target mean/std against raw training data only.
- Shared `train()` execution with `torch.nn.Linear`, validation, then held-out test evaluation in original target units.

The expected `raw`, `xscaler`, and `yscaler` keys follow the current test fixture and `src/symbolic.py` conventions. The data owner must confirm them before treating them as the production API. The script has not run against the stub.

## Colab notebook and exact commands

`notebooks/Group13_ANN_Experiments.ipynb` mounts Drive, syncs `phase-c-verification`, checks dependencies and hardware, validates Medium files and train-only target scaling, and runs pytest with skip reasons. Persistent data/results/checkpoints/figures are on Drive. Pilot and full experiments remain deferred.

After the owner changes are available in Colab, run from the repository root:

```python
import os
os.environ["SRSD_ROOT"] = "/content/drive/MyDrive/ann-group13/data/srsd-feynman_medium"
```

```bash
python -m pytest -q -rs tests/test_symbolic.py
python -m pytest -q -rs tests/test_data.py
python scripts/phase_c_data_smoke.py
python -m pytest -q -rs
```

`tests/test_data.py` is a requested new test module; it does not exist yet. The full suite must report exact passed, failed, and skipped counts for the branch head. Do not continue to Phase D while failures remain.

## Phase C blockers and ownership

1. Await Abdelrahman Elbehery's review/fix and regression tests for symbolic scaler round trips.
2. Await Jomana Mohamed's production loader and tests; then run the prepared smoke script.
3. Rerun the full suite in Colab on the current branch head and record exact counts.
4. Obtain the official group Phase 1 proposal and retain the team's/tutor's Medium dataset approval.
5. No branch commit or push has been made by this work yet. The current checkout is `phase-c-verification`; all additions in this commit are confined to this branch. Do not push to `master`.

**Status:** environment and dataset checks have prior Colab evidence; Phase C remains open because owner-owned modules are pending and current-head tests/smoke are unverified. Chebyshev KAN, runner, and aggregation implementation remain out of scope.

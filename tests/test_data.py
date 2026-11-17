"""Production data-loader tests using temporary files in the official split layout."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import load_problem

ROWS = {"train": 8000, "val": 1000, "test": 1000}


def write_problem(root: Path, d: int = 2) -> dict[str, np.ndarray]:
    raw = {}
    for split, n in ROWS.items():
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        # Validation/test values deliberately sit outside the training feature range.
        shift = 0.0 if split == "train" else (10000.0 if split == "val" else -10000.0)
        X = np.column_stack([
            np.linspace(1.0 + shift + i, 20.0 + shift + i, n, dtype=np.float64)
            for i in range(d)
        ])
        y = (X * np.arange(1, d + 1, dtype=np.float64)).sum(axis=1) + np.linspace(0, 1, n)
        arr = np.column_stack((X, y))
        np.savetxt(folder / "feynman-test.txt", arr)
        raw[split] = arr
    return raw


@pytest.mark.parametrize("d", range(1, 7))
def test_load_problem_shapes_finite_and_feature_dimensions(tmp_path, d):
    raw = write_problem(tmp_path, d)
    data = load_problem("feynman-test", root=tmp_path)
    assert data["d"] == d
    assert data["name"] == "feynman-test"
    assert set(data["raw"]) == set(ROWS)
    for split, short in (("train", "tr"), ("val", "va"), ("test", "te")):
        X, y = data[f"X{short}"], data[f"y{short}"]
        assert tuple(X.shape) == (ROWS[split], d)
        assert tuple(y.shape) == (ROWS[split], 1)
        assert X.dtype == torch.float32 and y.dtype == torch.float32
        assert torch.isfinite(X).all() and torch.isfinite(y).all()
        np.testing.assert_array_equal(data["raw"][split], raw[split])


def test_scalers_fit_training_split_only(tmp_path):
    raw = write_problem(tmp_path, d=2)
    data = load_problem("feynman-test", root=tmp_path)
    train_X, train_y = raw["train"][:, :2], raw["train"][:, 2]
    np.testing.assert_allclose(data["xscaler"].lo, train_X.min(axis=0))
    np.testing.assert_allclose(data["xscaler"].hi, train_X.max(axis=0))
    assert data["yscaler"].mean == pytest.approx(train_y.mean(dtype=np.float64))
    assert data["yscaler"].std == pytest.approx(train_y.std(dtype=np.float64))
    # Held-out outliers are transformed using fixed train bounds, not clipped/refitted.
    assert data["Xva"].abs().max().item() > 1
    assert data["Xte"].abs().max().item() > 1
    assert data["xscaler"].log_cols == frozenset()


def test_log_columns_use_train_fitted_log_bounds(tmp_path):
    raw = write_problem(tmp_path, d=2)
    # Keep held-out values positive so every split is valid for log10.
    for split in ("val", "test"):
        raw[split][:, 0] += 20000.0
        np.savetxt(tmp_path / split / "feynman-test.txt", raw[split])
    data = load_problem("feynman-test", root=tmp_path, log_cols=(0,))
    train_log = np.log10(data["raw"]["train"][:, 0])
    assert data["xscaler"].log_cols == frozenset({0})
    assert data["xscaler"].lo[0] == pytest.approx(train_log.min())
    assert data["xscaler"].hi[0] == pytest.approx(train_log.max())
    assert torch.isfinite(data["Xtr"]).all()


def test_invalid_equation_and_missing_file_are_clear(tmp_path):
    with pytest.raises(ValueError, match="Invalid equation name"):
        load_problem("../escape", root=tmp_path)
    for split in ROWS:
        (tmp_path / split).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Missing train data file"):
        load_problem("feynman-absent", root=tmp_path)


@pytest.mark.parametrize("split", ROWS)
def test_nonfinite_values_are_rejected_in_every_split(tmp_path, split):
    raw = write_problem(tmp_path, d=2)
    raw[split][0, 0] = np.nan
    np.savetxt(tmp_path / split / "feynman-test.txt", raw[split])
    with pytest.raises(ValueError, match="Non-finite value"):
        load_problem("feynman-test", root=tmp_path)


def test_log_preprocessing_rejects_nonpositive_values(tmp_path):
    raw = write_problem(tmp_path, d=2)
    raw["test"][0, 0] = 0.0
    np.savetxt(tmp_path / "test" / "feynman-test.txt", raw["test"])
    with pytest.raises(ValueError, match="non-positive"):
        load_problem("feynman-test", root=tmp_path, log_cols=(0,))


def test_wrong_split_row_count_is_rejected(tmp_path):
    for split, count in ROWS.items():
        folder = tmp_path / split
        folder.mkdir(parents=True, exist_ok=True)
        n = count - 1 if split == "val" else count
        arr = np.column_stack((np.linspace(1, 2, n), np.linspace(2, 4, n)))
        np.savetxt(folder / "feynman-test.txt", arr)
    with pytest.raises(ValueError, match="Malformed val data"):
        load_problem("feynman-test", root=tmp_path)

@pytest.mark.parametrize("bad_cols", [(2,), (-1,), (0.5,), (True,)])
def test_invalid_log_column_indices_are_rejected(tmp_path, bad_cols):
    write_problem(tmp_path, d=2)
    with pytest.raises(ValueError, match="log_cols"):
        load_problem("feynman-test", root=tmp_path, log_cols=bad_cols)


def test_missing_split_directory_is_reported(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="missing split directories"):
        load_problem("feynman-test", root=tmp_path)


def test_input_dimensions_must_match_across_splits(tmp_path):
    raw = write_problem(tmp_path, d=1)
    val = raw["val"]
    malformed_val = np.column_stack((val[:, 0], val[:, 0] ** 2, val[:, -1]))
    np.savetxt(tmp_path / "val" / "feynman-test.txt", malformed_val)
    with pytest.raises(ValueError, match="val feynman-test: expected 1 input columns"):
        load_problem("feynman-test", root=tmp_path)


def test_constant_training_input_is_rejected(tmp_path):
    raw = write_problem(tmp_path, d=1)
    raw["train"][:, 0] = 2.0
    np.savetxt(tmp_path / "train" / "feynman-test.txt", raw["train"])
    with pytest.raises(ValueError, match="input columns constant in training data"):
        load_problem("feynman-test", root=tmp_path)


def test_constant_training_target_is_rejected(tmp_path):
    raw = write_problem(tmp_path, d=2)
    raw["train"][:, -1] = 4.0
    np.savetxt(tmp_path / "train" / "feynman-test.txt", raw["train"])
    with pytest.raises(ValueError, match="target training mean/std are invalid"):
        load_problem("feynman-test", root=tmp_path)

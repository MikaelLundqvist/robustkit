import numpy as np
import pandas as pd
import pytest

from robustkit import (
    expand_aggregated_group_flat, expand_aggregated_group_borrowed_dispersion,
    expand_aggregated_table_flat, expand_aggregated_table_borrowed_dispersion,
    compare_reconstruction_methods,
)


def test_expand_aggregated_group_flat_has_zero_spread():
    values = expand_aggregated_group_flat(n=200, mean=40000)
    assert len(values) == 200
    assert np.all(values == 40000)
    assert np.std(values) == 0


def test_expand_aggregated_group_borrowed_dispersion_matches_target_ratio():
    values = expand_aggregated_group_borrowed_dispersion(n=50000, mean=40000, dispersion_ratio=0.45, seed=0)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    actual_ratio = (q3 - q1) / median
    assert actual_ratio == pytest.approx(0.45, abs=0.03)


def test_expand_aggregated_group_borrowed_dispersion_rejects_nonpositive_ratio():
    with pytest.raises(ValueError):
        expand_aggregated_group_borrowed_dispersion(n=10, mean=100, dispersion_ratio=-0.1)


def test_expand_aggregated_group_borrowed_dispersion_rejects_extreme_ratio():
    # A dispersion_ratio large enough to imply a negative Q1 should be rejected
    with pytest.raises(ValueError):
        expand_aggregated_group_borrowed_dispersion(n=10, mean=100, dispersion_ratio=3.0)


def test_expand_aggregated_table_flat_preserves_group_structure():
    df = pd.DataFrame({"age": ["25-29", "30-34"], "n": [100, 150], "mean_salary": [35000, 40000]})
    result = expand_aggregated_table_flat(df, n_col="n", mean_col="mean_salary", group_cols=["age"], value_name="salary")

    assert len(result) == 250
    assert result[result["age"] == "25-29"]["salary"].nunique() == 1
    assert result[result["age"] == "25-29"]["salary"].iloc[0] == 35000


def test_expand_aggregated_table_borrowed_dispersion_with_scalar_ratio():
    df = pd.DataFrame({"age": ["25-29", "30-34"], "n": [500, 500], "mean_salary": [35000, 40000]})
    result = expand_aggregated_table_borrowed_dispersion(
        df, n_col="n", mean_col="mean_salary", dispersion_ratio=0.4,
        group_cols=["age"], value_name="salary", seed=0,
    )
    assert len(result) == 1000
    assert result["salary"].std() > 0  # has actual spread, unlike the flat method


def test_expand_aggregated_table_borrowed_dispersion_with_per_row_ratio_column():
    df = pd.DataFrame({
        "age": ["25-29", "30-34"], "n": [500, 500],
        "mean_salary": [35000, 40000], "ratio": [0.3, 0.6],
    })
    result = expand_aggregated_table_borrowed_dispersion(
        df, n_col="n", mean_col="mean_salary", dispersion_ratio="ratio",
        group_cols=["age"], value_name="salary", seed=0,
    )
    std_low_ratio = result[result["age"] == "25-29"]["salary"].std()
    std_high_ratio = result[result["age"] == "30-34"]["salary"].std()
    assert std_high_ratio > std_low_ratio  # higher borrowed ratio -> more spread


def test_compare_reconstruction_methods_shows_flat_has_no_spread():
    result = compare_reconstruction_methods(n=5000, mean=52200, dispersion_ratio=0.45, seed=2)

    summary = result["summary"]
    flat_std = summary.loc[summary["method"] == "flat", "std"].iloc[0]
    borrowed_std = summary.loc[summary["method"] == "borrowed_dispersion", "std"].iloc[0]

    assert flat_std == 0
    assert borrowed_std > 0

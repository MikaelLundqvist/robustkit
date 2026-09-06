import numpy as np
import pandas as pd
import pytest

from robustkit import (
    expand_aggregated_group, expand_aggregated_table, check_reconstruction_quality,
    fit_huber_trend, predict_trend,
)


def test_expand_aggregated_group_reproduces_quantiles_at_large_n():
    values = expand_aggregated_group(n=50000, q1=30000, median=38000, q3=48000, seed=0)
    quality = check_reconstruction_quality(values, q1=30000, median=38000, q3=48000)

    assert quality["pct_error"]["q1"] < 2
    assert quality["pct_error"]["median"] < 2
    assert quality["pct_error"]["q3"] < 2


def test_expand_aggregated_group_degenerate_zero_spread():
    values = expand_aggregated_group(n=100, q1=40000, median=40000, q3=40000, seed=1)
    assert np.allclose(values, 40000, rtol=1e-9)


def test_expand_aggregated_group_rejects_invalid_ordering():
    with pytest.raises(ValueError):
        expand_aggregated_group(n=10, q1=50000, median=40000, q3=60000)


def test_expand_aggregated_group_rejects_non_positive_values():
    with pytest.raises(ValueError):
        expand_aggregated_group(n=10, q1=-5, median=100, q3=200)


def test_expand_aggregated_table_preserves_group_columns_and_size():
    df = pd.DataFrame({
        "year": [2020, 2021],
        "n": [500, 600],
        "q1": [30000, 31000],
        "median": [38000, 39000],
        "q3": [48000, 49500],
    })
    synthetic = expand_aggregated_table(
        df, n_col="n", q1_col="q1", median_col="median", q3_col="q3",
        group_cols=["year"], value_name="salary", seed=0,
    )

    assert len(synthetic) == 500 + 600
    assert set(synthetic["year"]) == {2020, 2021}
    assert (synthetic[synthetic["year"] == 2020].shape[0]) == 500
    assert (synthetic[synthetic["year"] == 2021].shape[0]) == 600


def test_expand_aggregated_table_reconstruction_recovers_known_trend():
    """
    End-to-end validation: reconstruct pseudo-individual data from a
    table of group summaries with a known underlying median trend, and
    confirm a Huber fit on the reconstructed data recovers that trend
    closely -- the core justification for this module's approach.
    """
    years = np.arange(2014, 2026)
    median_trend = 30000 + 1200 * (years - 2014)  # simple known linear trend
    df = pd.DataFrame({
        "year": years,
        "n": [3000] * len(years),
        "q1": median_trend * 0.8,
        "median": median_trend,
        "q3": median_trend * 1.3,
    })

    synthetic = expand_aggregated_table(
        df, n_col="n", q1_col="q1", median_col="median", q3_col="q3",
        group_cols=["year"], value_name="value", seed=42,
    )

    fit = fit_huber_trend(
        synthetic["year"].to_numpy(dtype=float), synthetic["value"].to_numpy(dtype=float), degree=2,
    )
    huber_pred = predict_trend(fit, years.astype(float))

    pct_diff = np.abs(huber_pred - median_trend) / median_trend * 100
    assert pct_diff.max() < 5


def test_check_reconstruction_quality_reports_expected_keys():
    values = expand_aggregated_group(n=1000, q1=30000, median=38000, q3=48000, seed=0)
    quality = check_reconstruction_quality(values, q1=30000, median=38000, q3=48000)

    assert set(quality.keys()) == {"target", "empirical", "pct_error"}
    assert set(quality["pct_error"].keys()) == {"q1", "median", "q3"}

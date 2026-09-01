"""
Smoke tests for robustkit.core. These check that every function runs
end-to-end and returns internally consistent output on synthetic data
-- not a full statistical validation suite, but enough to catch
breakage.
"""

import numpy as np
import pytest

from robustkit import (
    fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend,
    model_stability_pct, cooks_diagnostic, cook_impact,
    bootstrap_band, bca_bootstrap_ci,
    check_row_integrity, compare_row_sets,
)


@pytest.fixture
def xy_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(20, 60, 150)
    y = 1000 + 50 * x - 0.4 * x**2 + rng.normal(0, 300, 150)
    return x, y


def test_fit_and_predict_all_methods(xy_data):
    x, y = xy_data
    for fit_fn in (fit_huber_trend, fit_tukey_trend, fit_ols_trend):
        fit = fit_fn(x, y, degree=2)
        preds = predict_trend(fit, x_new=[25, 40, 55])
        assert len(preds) == 3
        assert np.all(np.isfinite(preds))


def test_model_stability(xy_data):
    x, y = xy_data
    result = model_stability_pct(x, y)
    assert result["median_pct_diff"] >= 0
    assert result["p95_pct_diff"] >= result["median_pct_diff"]
    assert result["max_pct_diff"] >= result["p95_pct_diff"]


def test_cooks_diagnostic_and_impact(xy_data):
    x, y = xy_data
    diag = cooks_diagnostic(x, y)
    assert len(diag["cooks_distance"]) == len(x)
    assert diag["threshold"] == pytest.approx(4 / len(x))

    if len(diag["flagged_indices"]) > 0:
        impact = cook_impact(x, y, diag["flagged_indices"])
        assert impact["n_excluded"] == len(diag["flagged_indices"])
        assert impact["median_pct_change"] >= 0


def test_cook_impact_with_injected_outlier():
    rng = np.random.default_rng(1)
    x = rng.uniform(20, 60, 100)
    y = 1000 + 50 * x + rng.normal(0, 100, 100)
    # Inject one severe influential point
    x = np.append(x, 90)
    y = np.append(y, 20000)

    diag = cooks_diagnostic(x, y)
    assert 100 in diag["flagged_indices"]  # the injected point (index 100)

    impact = cook_impact(x, y, diag["flagged_indices"])
    # Removing a genuinely influential point should change the curve
    # by a non-trivial amount somewhere along the grid.
    assert impact["max_pct_change"] > 1.0


def test_bootstrap_band(xy_data):
    x, y = xy_data
    band = bootstrap_band(x, y, n_boot=100)
    assert np.all(band["lower"] <= band["median"])
    assert np.all(band["median"] <= band["upper"])


def test_bca_bootstrap_ci(xy_data):
    x, y = xy_data
    result = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_), n_boot=200)
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_check_row_integrity():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "a", "b", None], "value": [1, 2, 3, 4]})
    result = check_row_integrity(df, "group")
    assert result["total_rows"] == 4
    assert result["rows_missing_group_label"] == 1
    assert result["rows_accounted_for"] == 3


def test_compare_row_sets():
    import pandas as pd
    before = pd.DataFrame({"id": [1, 2, 3, 4]})
    after = pd.DataFrame({"id": [1, 2, 3]})
    result = compare_row_sets(before, after, "id")
    assert result["n_dropped"] == 1
    assert result["n_added"] == 0
    assert result["dropped_keys"] == {4}

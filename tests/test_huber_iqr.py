import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from robustkit import plot_huber_iqr, fit_huber_trend, predict_trend


def make_concave_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(22, 65, n)
    salary = 25000 + 900 * (age - 22) - 12 * (age - 22) ** 2 + rng.normal(0, 800, n)
    return age, salary


def test_plot_huber_iqr_returns_expected_structure():
    age, salary = make_concave_data()
    result = plot_huber_iqr(age, salary, degree=2, bins=10)

    assert set(result.keys()) == {"grid", "huber_curve", "binned"}
    assert len(result["grid"]) == len(result["huber_curve"]) == 200
    assert (result["binned"]["q3"] >= result["binned"]["median"]).all()
    assert (result["binned"]["median"] >= result["binned"]["q1"]).all()


def test_plot_huber_iqr_curve_tracks_binned_medians():
    """
    The core claim this view exists to support: the Huber curve should
    closely track the actual per-bin medians, not diverge from them.
    """
    age, salary = make_concave_data()
    result = plot_huber_iqr(age, salary, degree=2, bins=10)

    fit = fit_huber_trend(age, salary, degree=2)
    huber_at_bin_centers = predict_trend(fit, result["binned"]["x_center"].to_numpy())
    pct_diff = np.abs(huber_at_bin_centers - result["binned"]["median"]) / result["binned"]["median"] * 100

    assert pct_diff.max() < 15


def test_plot_huber_iqr_show_points_defaults_to_false():
    import inspect
    sig = inspect.signature(plot_huber_iqr)
    assert sig.parameters["show_points"].default is False


def test_plot_huber_iqr_respects_bins_parameter():
    age, salary = make_concave_data()
    result_5 = plot_huber_iqr(age, salary, bins=5)
    result_15 = plot_huber_iqr(age, salary, bins=15)

    assert len(result_5["binned"]) <= 5
    assert len(result_15["binned"]) <= 15
    assert len(result_15["binned"]) > len(result_5["binned"])


def test_plot_huber_iqr_custom_title():
    import matplotlib.pyplot as plt

    age, salary = make_concave_data()
    custom_title = "Alla roller - med övertidsersättning (1000 personer)"
    plot_huber_iqr(age, salary, title=custom_title)
    assert plt.gca().get_title() == custom_title
    plt.close()


def test_plot_huber_iqr_default_title_when_not_specified():
    import matplotlib.pyplot as plt

    age, salary = make_concave_data()
    plot_huber_iqr(age, salary)
    assert plt.gca().get_title() == "Huber trend with per-bin median and IQR"
    plt.close()


def make_sparse_integer_age_data(n=200, seed=0):
    """Small n with integer ages -- guarantees some exact ages have
    very few observations, exercising the grouping='unique' NaN branch."""
    rng = np.random.default_rng(seed)
    age = rng.integers(22, 66, n).astype(float)
    salary = 25000 + 900 * (age - 22) - 12 * (age - 22) ** 2 + rng.normal(0, 1200, n)
    return age, salary


def test_plot_huber_iqr_unique_grouping_matches_exact_x_values():
    age, salary = make_sparse_integer_age_data()
    result = plot_huber_iqr(age, salary, grouping="unique", min_n_for_iqr=5)

    assert set(result["binned"]["x_center"]) <= set(age)


def test_plot_huber_iqr_unique_grouping_nan_for_sparse_n():
    age, salary = make_sparse_integer_age_data()
    result = plot_huber_iqr(age, salary, grouping="unique", min_n_for_iqr=5)

    binned = result["binned"]
    sparse = binned[binned["n"] < 5]
    dense = binned[binned["n"] >= 5]

    assert len(sparse) > 0 and len(dense) > 0  # sanity: this dataset should have both
    assert sparse["q1"].isna().all() and sparse["q3"].isna().all()
    assert dense["q1"].notna().all() and dense["q3"].notna().all()


def test_plot_huber_iqr_bin_grouping_never_produces_nan():
    age, salary = make_sparse_integer_age_data()
    result = plot_huber_iqr(age, salary, grouping="bin", bins=10)
    assert result["binned"]["q1"].notna().all()
    assert result["binned"]["q3"].notna().all()


def test_plot_huber_iqr_rejects_invalid_grouping():
    age, salary = make_concave_data()
    with pytest.raises(ValueError):
        plot_huber_iqr(age, salary, grouping="bogus")

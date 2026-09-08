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

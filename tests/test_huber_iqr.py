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

    assert set(result.keys()) == {"grid", "huber", "binned"}
    assert len(result["grid"]) == len(result["huber"]) == 200
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


# ---------------------------------------------------------------------------
# Block 2: multi-curve, style, bootstrap band, manual caps, MdAPE box
# ---------------------------------------------------------------------------

def test_plot_huber_iqr_multi_curve_huber_ols():
    """statsmodels-free subset of multi-curve support (tukey/median_ensemble need statsmodels)."""
    age, salary = make_concave_data()
    result = plot_huber_iqr(age, salary, methods=("huber", "ols"))
    assert set(result.keys()) == {"grid", "huber", "ols", "binned"}
    assert len(result["huber"]) == len(result["ols"]) == 200


def test_plot_huber_iqr_show_bootstrap_band_single_level():
    import matplotlib.pyplot as plt
    age, salary = make_concave_data()
    plot_huber_iqr(age, salary, show_bootstrap_band=True, bootstrap_levels=(95,))
    plt.close()


def test_plot_huber_iqr_show_bootstrap_band_dual_level():
    import matplotlib.pyplot as plt
    age, salary = make_concave_data()
    plot_huber_iqr(age, salary, show_bootstrap_band=True, bootstrap_levels=(95, 50))
    plt.close()


def test_plot_huber_iqr_manual_cap_style_runs():
    import matplotlib.pyplot as plt
    age, salary = make_sparse_integer_age_data()
    plot_huber_iqr(age, salary, cap_style="manual", grouping="unique", min_n_for_iqr=5)
    plt.close()


def test_plot_huber_iqr_mdape_residual_box():
    import matplotlib.pyplot as plt
    age, salary = make_concave_data()
    plot_huber_iqr(age, salary, residual_box_metric="mdape")
    plt.close()


def test_plot_huber_iqr_dynamic_ylim_sets_limits():
    import matplotlib.pyplot as plt
    age, salary = make_concave_data()
    plot_huber_iqr(age, salary, ylim="dynamic")
    lo, hi = plt.gca().get_ylim()
    assert lo < hi
    plt.close()


def test_plot_huber_iqr_style_override_applies():
    import matplotlib.pyplot as plt
    age, salary = make_concave_data()
    plot_huber_iqr(age, salary, methods=("huber",), style={"huber_line": {"color": "red", "linewidth": 3, "linestyle": "-", "label": "Huber poly(2)"}})
    line = [l for l in plt.gca().get_lines() if l.get_label() == "Huber poly(2)"][0]
    assert line.get_color() == "red"
    assert line.get_linewidth() == 3
    plt.close()


def test_plot_huber_iqr_rejects_invalid_residual_box_metric():
    age, salary = make_concave_data()
    with pytest.raises(ValueError):
        plot_huber_iqr(age, salary, residual_box_metric="bogus")


def test_plot_huber_iqr_rejects_invalid_cap_style():
    age, salary = make_concave_data()
    with pytest.raises(ValueError):
        plot_huber_iqr(age, salary, cap_style="bogus")


def test_plot_huber_iqr_backward_compatible_defaults_unchanged():
    """Regression: calling with only the original parameters must
    behave exactly as before this block's changes."""
    age, salary = make_concave_data()
    result = plot_huber_iqr(age, salary, degree=2, bins=10)
    fit = fit_huber_trend(age, salary, degree=2)
    direct = predict_trend(fit, result["grid"])
    assert np.allclose(result["huber"], direct)

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from robustkit import (
    pointwise_stability_report, plot_pointwise_stability, model_stability_pct, bootstrap_band,
)


@pytest.fixture
def sparse_young():
    """Few observations at low x, many elsewhere, with a 5% bonus contamination."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.uniform(0, 3, 40), rng.uniform(3, 35, 1200)])
    y = (32000 + 2200 * x - 40 * x ** 2) * np.exp(rng.normal(0, 0.06, len(x)))
    y = np.where(rng.random(len(x)) < 0.05, y * 1.6, y)
    return x, y


def test_columns_and_shape(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot=30, n_points=25)
    assert len(r) == 25
    assert list(r.columns) == ["x", "huber", "tukey", "ols", "method_spread_pct", "band_lower", "band_upper",
                               "band_width_pct", "local_n", "ratio", "dominant"]


def test_matches_underlying_functions(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot=30, n_points=25, seed=3)
    stab = model_stability_pct(x, y, n_points=25)
    band = bootstrap_band(x, y, n_boot=30, n_points=25, seed=3)
    assert np.allclose(r["x"], stab["grid"])
    assert np.allclose(r["method_spread_pct"], stab["pct_spread"])
    assert np.allclose(r["band_lower"], band["lower"])
    assert np.allclose(r["band_upper"], band["upper"])
    assert np.allclose(r["band_width_pct"], (band["upper"] - band["lower"]) / np.abs(stab["huber"]) * 100)


def test_local_n_counts_observations_in_window():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = 2 * x + 1 + np.array([0.1, -0.1] * 5 + [0.0])
    r = pointwise_stability_report(x, y, degree=1, n_boot=10, n_points=3, window_frac=0.2)
    # half window = 0.2 * 10 / 2 = 1.0 : grid 0, 5, 10 -> 2, 3, 2 observations
    assert list(r["local_n"]) == [2, 3, 2]
    assert r.attrs["window_half_width"] == pytest.approx(1.0)


def test_band_is_wider_where_data_are_sparse(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot=100, n_points=30)
    young = r[r["x"] < 3]
    middle = r[(r["x"] > 10) & (r["x"] < 25)]
    assert young["band_width_pct"].mean() > 2 * middle["band_width_pct"].mean()
    assert young["local_n"].mean() < middle["local_n"].mean()


def test_dominant_follows_comparison(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot=30, n_points=25)
    expected = np.where(r["band_width_pct"] > r["method_spread_pct"], "sampling", "method")
    assert list(r["dominant"]) == list(expected)
    assert set(r["dominant"]) <= {"sampling", "method"}


def test_ratio_is_spread_over_width(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot=30, n_points=25)
    assert np.allclose(r["ratio"], r["method_spread_pct"] / r["band_width_pct"])


def test_reproducible_with_seed_and_varies_without(sparse_young):
    x, y = sparse_young
    a = pointwise_stability_report(x, y, n_boot=30, n_points=10, seed=1)
    b = pointwise_stability_report(x, y, n_boot=30, n_points=10, seed=1)
    c = pointwise_stability_report(x, y, n_boot=30, n_points=10, seed=2)
    assert a.equals(b)
    assert not np.allclose(a["band_lower"], c["band_lower"])


def test_attrs_record_resolved_n_boot(sparse_young):
    x, y = sparse_young
    r = pointwise_stability_report(x, y, n_boot="auto", n_points=5, ci=90)
    assert r.attrs["n_boot"] == 500 and r.attrs["ci"] == 90


def test_input_validation():
    cases = [dict(x=[1, 2, 3], y=[1, 2]), dict(x=[1, 2], y=[1, 2]),
             dict(x=[1, 2, 3, np.nan], y=[1, 2, 3, 4])]
    for bad in cases:
        with pytest.raises(ValueError):
            pointwise_stability_report(bad["x"], bad["y"], n_boot=5)


def test_window_frac_validation(sparse_young):
    x, y = sparse_young
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="window_frac"):
            pointwise_stability_report(x, y, n_boot=5, window_frac=bad)


def test_plot_returns_axes_and_leaves_data_unchanged(sparse_young):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = sparse_young
    x0, y0 = x.copy(), y.copy()
    ax = plot_pointwise_stability(x, y, n_boot=20, n_points=15)
    assert hasattr(ax, "plot") and len(ax.lines) == 2
    ax_c = plot_pointwise_stability(x, y, kind="curves", n_boot=20, n_points=15)
    assert len(ax_c.lines) >= 4
    with pytest.raises(ValueError, match="kind"):
        plot_pointwise_stability(x, y, kind="nope", n_boot=20, n_points=15)
    assert np.array_equal(x, x0) and np.array_equal(y, y0)
    plt.close("all")

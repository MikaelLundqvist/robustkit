import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from robustkit import iqr, dispersion_ratio, dispersion_by_bin, plot_analyst_view, plot_publisher_view


def make_growing_dispersion_data(n=1000, seed=0):
    """Right-skewed data where spread genuinely grows with x (age)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(22, 65, n)
    base = 25000 + 500 * age
    noise_sigma = 0.05 + 0.003 * age
    salary = base * rng.lognormal(mean=0, sigma=noise_sigma)
    return age, salary


def test_iqr_and_dispersion_ratio_basic():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert iqr(y) == pytest.approx(4.0, abs=0.5)
    ratio = dispersion_ratio(y)
    assert ratio > 0


def test_dispersion_ratio_zero_median_returns_zero():
    y = np.array([-1, 0, 0, 0, 1])  # median is 0
    assert dispersion_ratio(y) == 0.0


def test_dispersion_by_bin_ordering_and_shape():
    age, salary = make_growing_dispersion_data()
    binned = dispersion_by_bin(age, salary, n_bins=8)

    assert len(binned) <= 8
    assert (binned["q3"] >= binned["median"]).all()
    assert (binned["median"] >= binned["q1"]).all()
    assert (binned["x_center"].diff().dropna() > 0).all()  # sorted ascending


def test_dispersion_by_bin_captures_growing_spread():
    age, salary = make_growing_dispersion_data()
    binned = dispersion_by_bin(age, salary, n_bins=8)
    assert binned["dispersion_ratio"].iloc[-1] > binned["dispersion_ratio"].iloc[0]


def test_analyst_view_ci_shrinks_with_more_data():
    """
    The core distinction this module exists to demonstrate: a
    confidence interval on the trend estimate shrinks as sample size
    grows, unlike population dispersion (see the publisher-view test
    below).
    """
    rng = np.random.default_rng(1)
    age, salary = make_growing_dispersion_data(n=500, seed=1)

    band_small = plot_analyst_view(age, salary, n_boot=150, show_points=False)
    width_small = band_small["upper"][25] - band_small["lower"][25]

    age_big = np.concatenate([age] * 20)
    salary_big = np.concatenate([salary] * 20) * (1 + rng.normal(0, 0.001, len(age) * 20))
    band_big = plot_analyst_view(age_big, salary_big, n_boot=100, show_points=False)
    width_big = band_big["upper"][25] - band_big["lower"][25]

    assert width_big < width_small


def test_publisher_view_iqr_does_not_shrink_with_more_data():
    """
    Contrast with the analyst-view test above: IQR reflects genuine
    population spread and should stay roughly constant regardless of
    sample size, for data drawn from the same underlying distribution.
    """
    rng = np.random.default_rng(2)
    age, salary = make_growing_dispersion_data(n=500, seed=2)

    binned_small = plot_publisher_view(age, salary, n_bins=8, show_points=False)

    age_big = np.concatenate([age] * 20)
    salary_big = np.concatenate([salary] * 20) * (1 + rng.normal(0, 0.001, len(age) * 20))
    binned_big = plot_publisher_view(age_big, salary_big, n_bins=8, show_points=False)

    iqr_small = (binned_small["q3"] - binned_small["q1"]).iloc[3]
    iqr_big = (binned_big["q3"] - binned_big["q1"]).iloc[3]

    assert abs(iqr_small - iqr_big) / iqr_small < 0.2


def test_publisher_view_show_points_defaults_to_false():
    import inspect
    sig = inspect.signature(plot_publisher_view)
    assert sig.parameters["show_points"].default is False

import numpy as np

from robustkit import (
    fit_huber_trend, predict_trend, trend_derivative,
    goodness_of_fit, compare_polynomial_degrees,
)


def make_quadratic_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(22, 65, n)
    salary = 25000 + 900 * (age - 22) - 12 * (age - 22) ** 2 + rng.normal(0, 800, n)
    return age, salary


def make_wavy_data(n=500, seed=0):
    """A trend with structure a quadratic cannot capture."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(22, 65, n)
    salary = (
        25000 + 900 * (age - 22) - 5 * (age - 22) ** 2
        + 8000 * np.sin((age - 22) / 6) + rng.normal(0, 800, n)
    )
    return age, salary


def test_trend_derivative_declines_for_concave_trend():
    age, salary = make_quadratic_data()
    fit = fit_huber_trend(age, salary, degree=2)

    rates = trend_derivative(fit, [25, 35, 45, 55, 65])
    assert rates[0] > rates[-1]
    assert np.all(np.diff(rates) < 0)  # monotonically declining for this concave trend


def test_trend_derivative_matches_predict_trend_direction():
    """A rising trend should have a positive derivative throughout its rising range."""
    age, salary = make_quadratic_data()
    fit = fit_huber_trend(age, salary, degree=2)

    rate_at_30 = trend_derivative(fit, [30])[0]
    assert rate_at_30 > 0  # still on the rising part of the concave trend


def test_goodness_of_fit_high_for_correctly_specified_degree():
    age, salary = make_quadratic_data()
    result = goodness_of_fit(age, salary, degree=2, method="huber")

    assert result["r_squared"] > 0.9
    assert result["rmse"] > 0
    assert result["mae"] > 0


def test_goodness_of_fit_lower_for_underspecified_degree():
    age, salary = make_quadratic_data()
    linear = goodness_of_fit(age, salary, degree=1, method="huber")
    quadratic = goodness_of_fit(age, salary, degree=2, method="huber")

    assert quadratic["r_squared"] > linear["r_squared"]


def test_compare_polynomial_degrees_stable_for_quadratic_data():
    """
    Regression test for a numerical-stability bug: degree-4/5 fits on
    raw (unscaled) polynomial features previously became unreliable
    and could score *worse* than degree-2 even on well-behaved data,
    due to ill-conditioning at high powers of realistic x values (e.g.
    age^5). x is now standardized before building polynomial features,
    so R^2 should stay stable (not collapse) at higher degrees.
    """
    age, salary = make_quadratic_data()
    comparison = compare_polynomial_degrees(age, salary, degrees=(1, 2, 3, 4), method="huber")

    degrees_2plus = comparison[comparison["degree"] >= 2]
    assert (degrees_2plus["r_squared"] > 0.9).all()
    assert degrees_2plus["r_squared"].std() < 0.01  # stable, not collapsing


def test_compare_polynomial_degrees_detects_underfit_quadratic():
    """
    For a genuinely non-quadratic (wavy) trend, higher polynomial
    degrees should meaningfully improve the fit -- confirming that a
    quadratic default isn't blindly assumed to be sufficient.
    """
    age, salary = make_wavy_data()
    comparison = compare_polynomial_degrees(age, salary, degrees=(1, 2, 3, 4, 5), method="huber")

    r2_at_2 = comparison.loc[comparison["degree"] == 2, "r_squared"].iloc[0]
    r2_at_5 = comparison.loc[comparison["degree"] == 5, "r_squared"].iloc[0]

    assert r2_at_5 > r2_at_2 + 0.1  # substantial improvement, not noise-level
    assert (comparison["r_squared"] >= -0.01).all()  # no numerical collapse into negative R^2

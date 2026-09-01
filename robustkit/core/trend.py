"""
Fit a trend curve -- y as a function of a single continuous x -- using
three different loss functions: Huber, Tukey biweight, and ordinary
least squares (OLS).

All three share the same polynomial design matrix, so their fitted
curves are directly comparable. OLS is included deliberately: if a
robust method and OLS agree closely, that agreement is itself useful
evidence that the conclusion is not driven by a handful of extreme
points (see robustkit.core.stability).

Note: statsmodels is only imported lazily, inside fit_tukey_trend and
the statsmodels branch of predict_trend. Huber and OLS fitting (via
scikit-learn) work without statsmodels installed at all.
"""

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def _design_matrix(x, degree):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(x), poly


def fit_huber_trend(x, y, degree=2, epsilon=1.35, alpha=0.0001):
    """
    Fit y ~ poly(x, degree) using Huber loss.

    epsilon controls the point at which the loss switches from
    quadratic to linear (smaller = more aggressive downweighting of
    outliers; 1.35 is the common default, tuned for ~95% efficiency
    under normal errors).
    """
    X, poly = _design_matrix(x, degree)
    model = HuberRegressor(epsilon=epsilon, alpha=alpha)
    model.fit(X, y)
    return {"model": model, "poly": poly, "kind": "sklearn"}


def fit_tukey_trend(x, y, degree=2, c=4.685):
    """
    Fit y ~ poly(x, degree) using Tukey's biweight (redescending) loss
    via statsmodels' robust linear model (RLM).

    Unlike Huber, Tukey's loss fully suppresses the influence of very
    extreme points rather than merely capping it -- useful when Huber
    still seems pulled by a handful of severe outliers.

    Requires statsmodels (imported lazily here).
    """
    import statsmodels.api as sm

    X, poly = _design_matrix(x, degree)
    X_sm = sm.add_constant(X)
    model = sm.RLM(np.asarray(y, dtype=float), X_sm, M=sm.robust.norms.TukeyBiweight(c=c)).fit()
    return {"model": model, "poly": poly, "kind": "statsmodels"}


def fit_ols_trend(x, y, degree=2):
    """
    Fit y ~ poly(x, degree) using ordinary least squares. Serves as a
    non-robust reference point, not a method to be discarded.
    """
    X, poly = _design_matrix(x, degree)
    model = LinearRegression()
    model.fit(X, y)
    return {"model": model, "poly": poly, "kind": "sklearn"}


def predict_trend(fit, x_new):
    """
    Predict y for new x values from a fit dict returned by
    fit_huber_trend / fit_tukey_trend / fit_ols_trend.
    """
    x_new = np.asarray(x_new, dtype=float).reshape(-1, 1)
    X_new = fit["poly"].transform(x_new)

    if fit["kind"] == "statsmodels":
        import statsmodels.api as sm
        X_new = sm.add_constant(X_new, has_constant="add")
        return np.asarray(fit["model"].predict(X_new))

    return np.asarray(fit["model"].predict(X_new))

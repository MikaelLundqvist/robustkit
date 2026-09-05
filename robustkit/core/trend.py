"""
Fit a trend curve -- y as a function of a single continuous x -- using
three different loss functions: Huber, Tukey biweight, and ordinary
least squares (OLS).

All three share the same polynomial design matrix, so their fitted
curves are directly comparable. OLS is included deliberately: if a
robust method and OLS agree closely, that agreement is itself useful
evidence that the conclusion is not driven by a handful of extreme
points (see robustkit.core.stability).

x is standardized before building polynomial features (see
_fit_design_matrix). Raw polynomial features (x, x^2, x^3, ...) become
severely ill-conditioned at realistic x scales -- e.g. age^5 for
age=65 exceeds a billion while age^1 is 65 -- which makes higher-degree
fits numerically unreliable (in testing, degree-4/5 Huber fits scored
*worse* than degree-2 on data that genuinely had more structure to
capture, purely from numerical instability, not overfitting).
Standardizing x first fixes this without changing what the fitted
curve represents back in the original x scale.

Note: statsmodels is only imported lazily, inside fit_tukey_trend and
the statsmodels branch of predict_trend. Huber and OLS fitting (via
scikit-learn) work without statsmodels installed at all.
"""

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _fit_design_matrix(x, degree):
    """
    Build a polynomial design matrix for FITTING: fits a StandardScaler
    on x, then expands the scaled x into polynomial features.

    Returns (X, poly, scaler). Both poly and scaler must be reused (via
    _transform_design_matrix) to build a matching design matrix for any
    new x values -- e.g. at prediction time.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(x_scaled)
    return X, poly, scaler


def _transform_design_matrix(x_new, poly, scaler):
    """Apply an already-fitted scaler + polynomial transform to new x values."""
    x_new = np.asarray(x_new, dtype=float).reshape(-1, 1)
    x_scaled = scaler.transform(x_new)
    return poly.transform(x_scaled)


def fit_huber_trend(x, y, degree=2, epsilon=1.35, alpha=0.0001):
    """
    Fit y ~ poly(x, degree) using Huber loss.

    epsilon controls the point at which the loss switches from
    quadratic to linear (smaller = more aggressive downweighting of
    outliers; 1.35 is the common default, tuned for ~95% efficiency
    under normal errors).
    """
    X, poly, scaler = _fit_design_matrix(x, degree)
    model = HuberRegressor(epsilon=epsilon, alpha=alpha)
    model.fit(X, y)
    return {"model": model, "poly": poly, "scaler": scaler, "kind": "sklearn"}


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

    X, poly, scaler = _fit_design_matrix(x, degree)
    X_sm = sm.add_constant(X)
    model = sm.RLM(np.asarray(y, dtype=float), X_sm, M=sm.robust.norms.TukeyBiweight(c=c)).fit()
    return {"model": model, "poly": poly, "scaler": scaler, "kind": "statsmodels"}


def fit_ols_trend(x, y, degree=2):
    """
    Fit y ~ poly(x, degree) using ordinary least squares. Serves as a
    non-robust reference point, not a method to be discarded.
    """
    X, poly, scaler = _fit_design_matrix(x, degree)
    model = LinearRegression()
    model.fit(X, y)
    return {"model": model, "poly": poly, "scaler": scaler, "kind": "sklearn"}


def predict_trend(fit, x_new):
    """
    Predict y for new x values from a fit dict returned by
    fit_huber_trend / fit_tukey_trend / fit_ols_trend.
    """
    X_new = _transform_design_matrix(x_new, fit["poly"], fit["scaler"])

    if fit["kind"] == "statsmodels":
        import statsmodels.api as sm
        X_new = sm.add_constant(X_new, has_constant="add")
        return np.asarray(fit["model"].predict(X_new))

    return np.asarray(fit["model"].predict(X_new))


def trend_derivative(fit, x, h=0.01):
    """
    Numerically differentiate a fitted trend at the given x values --
    the local RATE OF CHANGE (e.g. "salary growth per year of age")
    rather than the trend level itself.

    Uses a central difference, so this works generically for any fit
    dict from fit_huber_trend / fit_tukey_trend / fit_ols_trend,
    without needing to know the underlying model's functional form --
    it just calls predict_trend twice per point.

    h: step size for the central difference, in the ORIGINAL x scale
    (e.g. years of age) -- internal standardization is handled
    transparently by predict_trend. 0.01 is a reasonable default for
    most real-world x scales but can be tuned if x has a very
    different range.
    """
    x = np.asarray(x, dtype=float)
    y_plus = predict_trend(fit, x + h)
    y_minus = predict_trend(fit, x - h)
    return (y_plus - y_minus) / (2 * h)

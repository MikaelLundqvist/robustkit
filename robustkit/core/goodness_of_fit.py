"""
Measures of how well a fitted trend explains the variation in y, and a
diagnostic for choosing the polynomial degree empirically rather than
assuming a fixed degree fits every trend equally well.

Note on R² across methods: for Huber and Tukey fits, the R² computed
here is a descriptive "variance explained" measure derived from the
fit's predictions -- it is NOT the objective those methods actually
minimize (Huber/Tukey minimize a robust loss, not squared error). It
remains a useful, comparable summary of how much of y's variation the
fitted curve captures, consistently computed the same way across
methods and across polynomial degrees, which is what makes it useful
for comparison even though it isn't each method's "native" fit
statistic.
"""

import numpy as np
import pandas as pd

from .trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend

_FIT_FUNCTIONS = {
    "huber": fit_huber_trend,
    "tukey": fit_tukey_trend,
    "ols": fit_ols_trend,
}


def goodness_of_fit(x, y, degree=2, method="huber"):
    """
    Fit a trend and report how well it explains the variation in y.

    Returns r_squared (1 - var(residuals)/var(y)), plus RMSE and MAE
    of residuals -- reported alongside R² since a single R²-style
    number can hide whether errors are dominated by a few large misses
    or spread evenly across observations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fit_fn = _FIT_FUNCTIONS[method]
    fit = fit_fn(x, y, degree=degree)
    y_pred = predict_trend(fit, x)

    residuals = y - y_pred
    var_y = np.var(y)
    r_squared = 1 - np.var(residuals) / var_y if var_y > 0 else 0.0

    return {
        "method": method,
        "degree": degree,
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def compare_polynomial_degrees(x, y, degrees=(1, 2, 3, 4), method="huber"):
    """
    Fit the same (x, y) data at several polynomial degrees and report
    goodness_of_fit for each, so the right degree can be chosen
    empirically rather than assumed.

    A quadratic trend (the package default) is a reasonable starting
    point for many relationships, but not all -- some trends are more
    volatile and need a higher degree to be captured honestly, while
    an unnecessarily high degree risks fitting noise rather than a
    real pattern. The r_squared_gain column shows how much each extra
    degree actually buys you: a large gain going from degree 2 to 3
    suggests the quadratic default is too restrictive for this trend;
    a negligible gain suggests the extra complexity isn't earning its
    keep.
    """
    rows = [goodness_of_fit(x, y, degree=d, method=method) for d in degrees]
    df = pd.DataFrame(rows)
    df["r_squared_gain"] = df["r_squared"].diff()
    return df

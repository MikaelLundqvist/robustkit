"""
Quantify how much a fitted trend curve changes depending on which
fitting method (Huber, Tukey, or OLS) is used.

The central idea: a curve shape that is nearly identical across all
three methods is a more trustworthy conclusion than one that only
appears under a single method.
"""

import numpy as np

from .trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend


def model_stability_pct(x, y, degree=2, n_points=50):
    """
    Fit Huber, Tukey, and OLS trends to the same (x, y) data, evaluate
    all three on a common grid, and report how far apart they are as a
    percentage of the average predicted value at each grid point.

    Returns a dict with the three predicted curves plus summary
    statistics (median / 95th percentile / max percentage spread).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.linspace(x.min(), x.max(), n_points)

    huber_fit = fit_huber_trend(x, y, degree=degree)
    tukey_fit = fit_tukey_trend(x, y, degree=degree)
    ols_fit = fit_ols_trend(x, y, degree=degree)

    huber_pred = predict_trend(huber_fit, grid)
    tukey_pred = predict_trend(tukey_fit, grid)
    ols_pred = predict_trend(ols_fit, grid)

    curves = np.vstack([huber_pred, tukey_pred, ols_pred])
    spread = curves.max(axis=0) - curves.min(axis=0)
    avg = np.abs(curves.mean(axis=0))
    avg[avg == 0] = np.nan  # avoid divide-by-zero; results in NaN, not inf
    pct_spread = spread / avg * 100

    return {
        "grid": grid,
        "huber": huber_pred,
        "tukey": tukey_pred,
        "ols": ols_pred,
        "pct_spread": pct_spread,
        "median_pct_diff": float(np.nanmedian(pct_spread)),
        "p95_pct_diff": float(np.nanpercentile(pct_spread, 95)),
        "max_pct_diff": float(np.nanmax(pct_spread)),
    }

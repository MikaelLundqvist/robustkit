"""
Identify individually influential observations, and quantify how much
they actually change the fitted trend if removed.

cooks_diagnostic() answers "which points stand out?"
cook_impact() answers "how much does it matter if we remove them?" --
these are deliberately separate steps, since a flagged point is not
automatically a point worth acting on.

Cook's distance is computed here via plain OLS and the hat matrix
(pure numpy/scikit-learn), so this module has no statsmodels
dependency.
"""

import numpy as np

from .trend import _fit_design_matrix, fit_huber_trend, predict_trend


def cooks_diagnostic(x, y, degree=2):
    """
    Compute Cook's distance for each observation, based on an OLS fit
    of y ~ poly(x, degree). Flags points above the conventional 4/n
    threshold.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    X, _, _ = _fit_design_matrix(x, degree)
    X_design = np.column_stack([np.ones(n), X])
    p = X_design.shape[1]

    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    residuals = y - y_hat

    mse = np.sum(residuals ** 2) / (n - p)

    Q, R = np.linalg.qr(X_design)
    leverage = np.sum(Q ** 2, axis=1)
    leverage = np.clip(leverage, 1e-12, 1 - 1e-12)

    cooks_d = (residuals ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)

    threshold = 4 / n
    flagged_indices = np.where(cooks_d > threshold)[0]

    return {
        "cooks_distance": cooks_d,
        "leverage": leverage,
        "threshold": threshold,
        "flagged_indices": flagged_indices,
    }


def cook_impact(x, y, flagged_indices, degree=2, n_points=50):
    """
    Compare the Huber-fitted trend curve with and without the given
    flagged observations, to quantify how much they actually pull the
    curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.linspace(x.min(), x.max(), n_points)

    full_fit = fit_huber_trend(x, y, degree=degree)
    full_pred = predict_trend(full_fit, grid)

    mask = np.ones(len(x), dtype=bool)
    mask[np.asarray(flagged_indices, dtype=int)] = False

    if mask.sum() < degree + 2:
        raise ValueError(
            "Too few remaining observations to fit a comparison curve "
            "after excluding flagged points."
        )

    reduced_fit = fit_huber_trend(x[mask], y[mask], degree=degree)
    reduced_pred = predict_trend(reduced_fit, grid)

    denom = np.abs(full_pred)
    denom[denom == 0] = np.nan
    pct_change = np.abs(full_pred - reduced_pred) / denom * 100

    return {
        "grid": grid,
        "with_flagged": full_pred,
        "without_flagged": reduced_pred,
        "pct_change": pct_change,
        "median_pct_change": float(np.nanmedian(pct_change)),
        "max_pct_change": float(np.nanmax(pct_change)),
        "n_excluded": int((~mask).sum()),
    }

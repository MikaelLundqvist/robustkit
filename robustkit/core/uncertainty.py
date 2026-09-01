"""
Quantify uncertainty in a fitted trend curve (or any statistic derived
from (x, y)) via bootstrap resampling.

bootstrap_band() gives a simple percentile-based confidence band around
a Huber-fitted trend curve.

bca_bootstrap_ci() gives a bias-corrected and accelerated (BCa)
confidence interval for an arbitrary scalar statistic. BCa corrects for
both bias and skew in the bootstrap distribution, which plain percentile
bootstrap does not -- worth the extra computation when the underlying
distribution is noticeably skewed (e.g. right-skewed salary or price
data).
"""

import numpy as np
from scipy import stats

from .trend import fit_huber_trend, predict_trend


def bootstrap_band(x, y, degree=2, n_boot=500, ci=95, n_points=50, seed=0):
    """
    Percentile bootstrap confidence band for a Huber-fitted trend curve.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    grid = np.linspace(x.min(), x.max(), n_points)

    preds = np.empty((n_boot, n_points))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        fit = fit_huber_trend(x[idx], y[idx], degree=degree)
        preds[b] = predict_trend(fit, grid)

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct

    return {
        "grid": grid,
        "lower": np.percentile(preds, lower_pct, axis=0),
        "median": np.percentile(preds, 50, axis=0),
        "upper": np.percentile(preds, upper_pct, axis=0),
    }


def bca_bootstrap_ci(x, y, statistic_fn, n_boot=1000, ci=95, seed=0):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval
    for an arbitrary statistic computed from (x, y).

    statistic_fn: callable(x, y) -> float

    Returns the point estimate, the BCa interval bounds, and the
    bias-correction (z0) and acceleration (a) parameters, in case you
    want to inspect how much correction was actually applied.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    theta_hat = statistic_fn(x, y)

    boot_thetas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_thetas[b] = statistic_fn(x[idx], y[idx])

    # Bias correction: how far off-center is theta_hat within the
    # bootstrap distribution?
    prop_less = np.mean(boot_thetas < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)  # avoid +/- inf
    z0 = stats.norm.ppf(prop_less)

    # Acceleration via jackknife (leave-one-out)
    jack_thetas = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_thetas[i] = statistic_fn(x[mask], y[mask])

    jack_mean = jack_thetas.mean()
    num = np.sum((jack_mean - jack_thetas) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_thetas) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha = (100 - ci) / 100 / 2
    z_lo = stats.norm.ppf(alpha)
    z_hi = stats.norm.ppf(1 - alpha)

    def _adjust(z):
        denom = 1 - a * (z0 + z)
        adjusted_z = z0 + (z0 + z) / denom if denom != 0 else z0
        return stats.norm.cdf(adjusted_z)

    lo_pct = np.clip(_adjust(z_lo) * 100, 0, 100)
    hi_pct = np.clip(_adjust(z_hi) * 100, 0, 100)

    return {
        "estimate": theta_hat,
        "lower": float(np.percentile(boot_thetas, lo_pct)),
        "upper": float(np.percentile(boot_thetas, hi_pct)),
        "z0": float(z0),
        "a": float(a),
        "boot_distribution": boot_thetas,
    }

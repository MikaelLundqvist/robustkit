"""
Quantify uncertainty in a fitted trend curve (or any statistic derived
from (x, y)) via bootstrap resampling.
"""

import numpy as np
from scipy import stats

from .trend import fit_huber_trend, predict_trend


def bootstrap_band(x, y, degree=2, n_boot=500, ci=95, n_points=50, seed=0):
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
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    theta_hat = statistic_fn(x, y)

    boot_thetas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_thetas[b] = statistic_fn(x[idx], y[idx])

    prop_less = np.mean(boot_thetas < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

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

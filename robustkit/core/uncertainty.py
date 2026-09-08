"""
Quantify uncertainty in a fitted trend curve (or any statistic derived
from (x, y)) via bootstrap resampling.
"""

import numpy as np
from scipy import stats

from .trend import fit_huber_trend, predict_trend


def _resolve_n_boot(n, n_boot):
    """
    Resolve the "auto" n_boot schedule: bootstrap_band refits a full
    Huber model on every iteration, which becomes expensive at scale
    (n_boot=200 x ~54,000 rows was observed to make validation stall
    on a large real dataset). Reducing n_boot for large n trades a
    little bootstrap-band precision for tractable runtime; pass an
    explicit integer instead of "auto" to opt out of this and always
    get exactly that many iterations, regardless of n.
    """
    if n_boot != "auto":
        return n_boot
    if n > 20000:
        return 50
    if n > 10000:
        return 100
    if n > 5000:
        return 200
    return 500


def bootstrap_band(x, y, degree=2, n_boot="auto", ci=95, n_points=50, seed=0):
    """
    Percentile bootstrap confidence band for a Huber-fitted trend curve.

    n_boot: "auto" (default) scales the number of bootstrap iterations
        down as n grows, since each iteration refits a full Huber
        model (see _resolve_n_boot for the schedule). Pass an explicit
        integer to always use exactly that many iterations regardless
        of dataset size.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n_boot = _resolve_n_boot(n, n_boot)
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


def bca_bootstrap_ci_by_index(n, statistic_fn, n_boot=1000, ci=95, seed=0):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval
    for an arbitrary statistic, expressed as a function of RESAMPLED
    INDICES rather than resampled (x, y) arrays directly.

    This generalizes bca_bootstrap_ci (below) to statistics that need
    more than two aligned 1D arrays -- e.g. a benchmark model that
    predicts from several DataFrame columns at once (age, level,
    overtime status, cluster, ...), not just a single x. statistic_fn
    only needs to accept an array of integer indices (with
    replacement) into the original data of length n, and look up
    whatever columns/values it needs itself, e.g.:

        def statistic_fn(idx):
            sub = df.iloc[idx]
            return float(np.median(sub["y"] - my_model.predict(sub)))

    bca_bootstrap_ci itself is just a thin wrapper around this:
    statistic_fn(idx) = original_statistic_fn(x[idx], y[idx]).
    """
    rng = np.random.default_rng(seed)
    idx_full = np.arange(n)

    theta_hat = statistic_fn(idx_full)

    boot_thetas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_thetas[b] = statistic_fn(idx)

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
        jack_thetas[i] = statistic_fn(idx_full[mask])

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


def bca_bootstrap_ci(x, y, statistic_fn, n_boot=1000, ci=95, seed=0):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval
    for an arbitrary statistic computed from (x, y).

    statistic_fn: callable(x, y) -> float

    Returns the point estimate, the BCa interval bounds, and the
    bias-correction (z0) and acceleration (a) parameters, in case you
    want to inspect how much correction was actually applied.

    Implemented as a thin wrapper around bca_bootstrap_ci_by_index --
    see that function if you need a statistic that depends on more
    than two aligned arrays (e.g. several DataFrame columns).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    def by_index(idx):
        return statistic_fn(x[idx], y[idx])

    return bca_bootstrap_ci_by_index(n, by_index, n_boot=n_boot, ci=ci, seed=seed)

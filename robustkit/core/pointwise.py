"""
Where along x is a fitted trend weak?

model_stability_pct summarises how far Huber, Tukey and OLS curves are
apart over the whole x range, and bootstrap_band gives the sampling
uncertainty of the Huber curve at each x. Both are pointwise
underneath, but they answer different questions and are reported
separately. This module puts them side by side at every grid point,
together with how many observations sit near that point, so a region
of the curve can be classified as

    "sampling" -- the bootstrap band is wider than the methods'
                  spread: the data are too thin or noisy there to
                  say more, and the choice of method is not the issue;
    "method"   -- the methods differ by more than the bootstrap band:
                  the conclusion there depends on the fitting method.
"""

import numpy as np
import pandas as pd

from .stability import model_stability_pct
from .uncertainty import bootstrap_band, _resolve_n_boot


def pointwise_stability_report(x, y, degree=2, n_boot="auto", ci=95, n_points=50,
                               window_frac=0.1, seed=0):
    """
    Compare, at each point of a common grid over x, the spread between
    the Huber, Tukey and OLS curves with the bootstrap band of the
    Huber curve, and count the observations near that point.

    Returns a DataFrame with one row per grid point:

      x                   the grid point
      huber, tukey, ols   the three fitted curves at x
      method_spread_pct   max - min of the three curves as a percentage
                          of their mean (the pct_spread of
                          model_stability_pct)
      band_lower, band_upper
                          the percentile bootstrap band of the Huber
                          curve (see bootstrap_band)
      band_width_pct      (band_upper - band_lower) as a percentage of
                          the Huber curve at x; NaN where that is 0
      local_n             observations within +- half a window of x,
                          where the window is window_frac of the x
                          range -- a direct measure of how sparse the
                          data are around x
      ratio               method_spread_pct / band_width_pct; NaN where
                          the band has zero width
      dominant            "sampling" where band_width_pct is the larger
                          of the two, otherwise "method"

    n_boot, ci, n_points and seed are passed to bootstrap_band; n_points
    is shared with model_stability_pct so that the two use the same
    grid. The resolved number of bootstrap iterations is stored in
    report.attrs["n_boot"], together with "ci" and "window_half_width".

    Requires statsmodels (Tukey), like model_stability_pct.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional and of equal length; "
                         f"got shapes {x.shape} and {y.shape}")
    if len(x) < degree + 2:
        raise ValueError(f"need at least degree + 2 = {degree + 2} observations; got {len(x)}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must not contain NaN or infinite values")
    if not (0 < window_frac <= 1):
        raise ValueError(f"window_frac must be in (0, 1]; got {window_frac}")

    stability = model_stability_pct(x, y, degree=degree, n_points=n_points)
    band = bootstrap_band(x, y, degree=degree, n_boot=n_boot, ci=ci, n_points=n_points, seed=seed)
    grid = stability["grid"]
    assert np.allclose(grid, band["grid"])

    huber = stability["huber"]
    safe = np.where(huber == 0, np.nan, np.abs(huber))
    band_width_pct = (band["upper"] - band["lower"]) / safe * 100
    spread = stability["pct_spread"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(band_width_pct > 0, spread / band_width_pct, np.nan)

    half = window_frac * (x.max() - x.min()) / 2
    local_n = (np.abs(x[None, :] - grid[:, None]) <= half).sum(axis=1)

    dominant = np.where(band_width_pct > spread, "sampling", "method")
    report = pd.DataFrame({
        "x": grid, "huber": huber, "tukey": stability["tukey"], "ols": stability["ols"],
        "method_spread_pct": spread, "band_lower": band["lower"], "band_upper": band["upper"],
        "band_width_pct": band_width_pct, "local_n": local_n, "ratio": ratio, "dominant": dominant,
    })
    report.attrs.update({"n_boot": _resolve_n_boot(len(x), n_boot), "ci": ci, "window_half_width": float(half)})
    return report


def plot_pointwise_stability(x, y, report=None, kind="percent", ax=None, figsize=(9, 5), **report_kwargs):
    """
    Plot the output of pointwise_stability_report. Returns the matplotlib Axes.

    kind="percent" (default): the methods' spread and the bootstrap band's
        width, both as a percentage of the fitted value, against x, with the
        number of observations near each point (local_n) as bars on a second
        axis. This is the view that shows where a conclusion is limited by
        thin data (the band width rises while local_n falls) and where it is
        limited by the choice of method (the spread is the higher line).
    kind="curves": the three fitted curves and the Huber bootstrap band,
        with a rug of the observations. On the scale of the fitted values
        the differences are often too small to see, which is why "percent"
        is the default.

    Provide a precomputed `report`, or leave it out and it is computed from
    (x, y) with report_kwargs passed through.
    """
    import matplotlib.pyplot as plt

    if kind not in ("percent", "curves"):
        raise ValueError(f"kind must be 'percent' or 'curves'; got {kind!r}")
    x = np.asarray(x, dtype=float)
    if report is None:
        report = pointwise_stability_report(x, y, **report_kwargs)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ci = report.attrs.get("ci", 95)

    if kind == "percent":
        step = report["x"].diff().median()
        ax2 = ax.twinx()
        ax2.bar(report["x"], report["local_n"], width=step * 0.9, color="#d9d9d9", zorder=0,
                label="observations nearby (right axis)")
        ax2.set_ylabel("observations nearby")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(report["x"], report["method_spread_pct"], color="#d95f02", lw=2,
                label="spread between Huber, Tukey and OLS")
        ax.plot(report["x"], report["band_width_pct"], color="#08519c", lw=2,
                label=f"width of the {ci}% bootstrap band")
        ax.set_ylabel("% of fitted value")
        ax.set_xlabel("x")
        ax.set_title("Pointwise stability: methods vs sampling uncertainty")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
        ax.grid(True, alpha=0.3)
        return ax

    ax.fill_between(report["x"], report["band_lower"], report["band_upper"], color="#9ecae1", alpha=0.6,
                    label=f"Huber, {ci}% bootstrap band")
    ax.plot(report["x"], report["huber"], color="#08519c", lw=2, label="Huber")
    ax.plot(report["x"], report["tukey"], color="#d95f02", lw=1.5, ls="--", label="Tukey")
    ax.plot(report["x"], report["ols"], color="#1b9e77", lw=1.5, ls=":", label="OLS")
    lo, hi = ax.get_ylim()
    ax.plot(x, np.full_like(x, lo), "|", color="black", alpha=0.25, markersize=8, label="observations")
    ax.set_ylim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel("fitted y")
    ax.set_title("Pointwise stability: fitted curves")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax

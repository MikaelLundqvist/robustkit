"""
Combined view: a global Huber trend curve overlaid with median + IQR
error bars, plus an optional residual-quality box (R^2, MAE, RMSE) --
reproducing the main diagram style used in the wage-analysis workbook
this package is based on, where the fitted model, the actual
population spread, and a residual-quality indicator are all shown
together rather than as separate plots.

Distinguishes itself from the other two report views:
    - plot_analyst_view shows a bootstrap CONFIDENCE band around the
      fitted curve (narrows as n grows) -- uncertainty in the estimate.
    - plot_publisher_view shows median + IQR from binned data ALONE,
      with no fitted model curve at all -- population spread only.

plot_huber_iqr shows both together: the smooth Huber curve as the
model's summary of the trend, and the median + IQR as what the data
actually looks like -- so a reader can see at a glance how well the
smooth trend tracks the medians, and how wide the underlying spread is
at each point.

Two grouping strategies are supported (see `grouping`):
    - "bin" (default): quantile-based binning via dispersion_by_bin.
      Gives stable quantile estimates even in small populations, and
      works for any continuous x, not just one with few distinct
      values.
    - "unique": group by each EXACT x value (e.g. each individual age
      in years), matching a workbook-style groupby(x) aggregation.
      When a given x value has fewer than `min_n_for_iqr` observations,
      its IQR is not shown at all -- the ABSENCE of an error bar is
      itself information, signaling the sample at that exact value is
      too small to say anything about spread, rather than a plotting
      simplification.
"""

import numpy as np
import pandas as pd

from ..core.trend import fit_huber_trend, predict_trend
from ..core.goodness_of_fit import goodness_of_fit
from .dispersion import dispersion_by_bin


def _unique_value_summary(x, y, min_n_for_iqr=5):
    """
    Median/Q1/Q3 grouped by each EXACT x value (not binned). Q1/Q3 are
    NaN for any x value with fewer than min_n_for_iqr observations --
    intentionally, so the caller can distinguish "spread is small" from
    "spread is unknown because the sample is too small to say."
    """
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})

    rows = []
    for x_val, group in df.groupby("x"):
        n = len(group)
        median = float(group["y"].median())
        if n >= min_n_for_iqr:
            q1 = float(group["y"].quantile(0.25))
            q3 = float(group["y"].quantile(0.75))
        else:
            q1 = np.nan
            q3 = np.nan
        rows.append({"x_center": x_val, "n": n, "q1": q1, "median": median, "q3": q3})

    return pd.DataFrame(rows).sort_values("x_center").reset_index(drop=True)


def plot_huber_iqr(x, y, degree=2, bins=15, grouping="bin", min_n_for_iqr=5,
                    show_points=False, show_residual_box=True,
                    title=None, ax=None, figsize=(10, 6)):
    """
    Plot a Huber trend curve together with median + IQR error bars.

    grouping: "bin" (default) or "unique" -- see module docstring.
        `bins` is only used when grouping="bin"; `min_n_for_iqr` is
        only used when grouping="unique".
    show_points: off by default (consistent with plot_publisher_view),
        since this view is often used for potentially sensitive data
        (e.g. salaries) where individual points shouldn't be exposed
        without a deliberate choice to do so.
    show_residual_box: if True, annotate the plot with R^2, MAE, and
        RMSE of the Huber fit (via goodness_of_fit), as a quick
        indicator of how well the model summarizes the data -- not a
        substitute for compare_polynomial_degrees or
        model_stability_pct, just a glanceable summary on the plot
        itself.
    title: optional custom title, e.g. naming the population and
        filters shown (f"All roles -- overtime eligible ({len(df)}
        people)") so the chart is self-explanatory when used directly
        in a report or presentation, without extra context. Defaults
        to a generic title if omitted.

    Returns a dict with the fitted curve (`grid`, `huber_curve`) and
    the summary DataFrame (`binned` -- from dispersion_by_bin or
    _unique_value_summary depending on `grouping`) for further
    inspection. In "unique" mode, rows with n < min_n_for_iqr have
    q1/q3 = NaN.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fit = fit_huber_trend(x, y, degree=degree)
    grid = np.linspace(x.min(), x.max(), 200)
    huber_curve = predict_trend(fit, grid)

    if grouping == "bin":
        binned = dispersion_by_bin(x, y, n_bins=bins)
    elif grouping == "unique":
        binned = _unique_value_summary(x, y, min_n_for_iqr=min_n_for_iqr)
    else:
        raise ValueError(f"grouping must be 'bin' or 'unique', got {grouping!r}")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.scatter(x, y, s=10, alpha=0.15, color="gray", zorder=1)

    has_iqr = binned["q1"].notna() & binned["q3"].notna()
    with_iqr = binned[has_iqr]
    without_iqr = binned[~has_iqr]

    if len(with_iqr) > 0:
        yerr_lower = with_iqr["median"] - with_iqr["q1"]
        yerr_upper = with_iqr["q3"] - with_iqr["median"]
        iqr_label = (
            f"Median (IQR error bars, n\u2265{min_n_for_iqr})" if grouping == "unique"
            else "Median per bin (IQR error bars)"
        )
        ax.errorbar(
            with_iqr["x_center"], with_iqr["median"],
            yerr=[yerr_lower, yerr_upper],
            fmt="o", color="darkorange", ecolor="darkorange", elinewidth=1.5, capsize=4,
            label=iqr_label, zorder=3,
        )

    if len(without_iqr) > 0:
        ax.scatter(
            without_iqr["x_center"], without_iqr["median"],
            marker="o", facecolors="none", edgecolors="darkorange", s=40,
            label=f"Median (n<{min_n_for_iqr}, spread not shown)", zorder=3,
        )

    ax.plot(grid, huber_curve, color="steelblue", linewidth=2, label="Huber trend", zorder=2)

    if show_residual_box:
        gof = goodness_of_fit(x, y, degree=degree, method="huber")
        textstr = f"R\u00b2={gof['r_squared']:.2f}\nMAE={gof['mae']:.0f}\nRMSE={gof['rmse']:.0f}"
        ax.text(
            0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_title(title if title is not None else "Huber trend with per-bin median and IQR")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return {"grid": grid, "huber_curve": huber_curve, "binned": binned}

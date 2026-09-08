"""
Combined view: a global Huber trend curve overlaid with per-bin median
+ IQR error bars, plus an optional residual-quality box (R^2, MAE,
RMSE) -- reproducing the main diagram style used in the wage-analysis
workbook this package is based on, where the fitted model, the actual
population spread, and a residual-quality indicator are all shown
together rather than as separate plots.

Distinguishes itself from the other two report views:
    - plot_analyst_view shows a bootstrap CONFIDENCE band around the
      fitted curve (narrows as n grows) -- uncertainty in the estimate.
    - plot_publisher_view shows median + IQR from binned data ALONE,
      with no fitted model curve at all -- population spread only.

plot_huber_iqr shows both together: the smooth Huber curve as the
model's summary of the trend, and the per-bin median + IQR as what the
data actually looks like -- so a reader can see at a glance how well
the smooth trend tracks the binned medians, and how wide the
underlying spread is at each point.
"""

import numpy as np

from ..core.trend import fit_huber_trend, predict_trend
from ..core.goodness_of_fit import goodness_of_fit
from .dispersion import dispersion_by_bin


def plot_huber_iqr(x, y, degree=2, bins=15, show_points=False, show_residual_box=True,
                    ax=None, figsize=(10, 6)):
    """
    Plot a Huber trend curve together with per-bin median + IQR error
    bars.

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

    Returns a dict with the fitted curve (`grid`, `huber_curve`) and
    the per-bin summary DataFrame (`binned`, the same shape as
    dispersion_by_bin's output) for further inspection.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fit = fit_huber_trend(x, y, degree=degree)
    grid = np.linspace(x.min(), x.max(), 200)
    huber_curve = predict_trend(fit, grid)

    binned = dispersion_by_bin(x, y, n_bins=bins)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.scatter(x, y, s=10, alpha=0.15, color="gray", zorder=1)

    yerr_lower = binned["median"] - binned["q1"]
    yerr_upper = binned["q3"] - binned["median"]
    ax.errorbar(
        binned["x_center"], binned["median"],
        yerr=[yerr_lower, yerr_upper],
        fmt="o", color="darkorange", ecolor="darkorange", elinewidth=1.5, capsize=4,
        label="Median per bin (IQR error bars)", zorder=3,
    )

    ax.plot(grid, huber_curve, color="steelblue", linewidth=2, label="Huber trend", zorder=2)

    if show_residual_box:
        gof = goodness_of_fit(x, y, degree=degree, method="huber")
        textstr = f"R\u00b2={gof['r_squared']:.2f}\nMAE={gof['mae']:.0f}\nRMSE={gof['rmse']:.0f}"
        ax.text(
            0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_title("Huber trend with per-bin median and IQR")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return {"grid": grid, "huber_curve": huber_curve, "binned": binned}

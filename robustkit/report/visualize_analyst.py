"""
Analyst-mode visualization: a fitted trend plus the statistical
uncertainty around it -- a confidence band that narrows toward zero
width as sample size grows. Answers "how confident are we in this
estimate?", not "how spread out are the underlying values?" (see
visualize_publisher for the latter).
"""

import numpy as np

from ..core.uncertainty import bootstrap_band


def plot_analyst_view(x, y, degree=2, n_boot=500, ci=95, show_points=True, ax=None, figsize=(10, 6)):
    """
    Plot the Huber-fitted trend with a bootstrap confidence band.

    This band shrinks as sample size grows -- it describes uncertainty
    in the *estimate*, not dispersion in the population. Contrast with
    plot_publisher_view.report's IQR band, which reflects real spread
    and does not shrink with more data.

    Returns the bootstrap_band() result dict for further inspection.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    band = bootstrap_band(x, y, degree=degree, n_boot=n_boot, ci=ci)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.scatter(x, y, s=15, alpha=0.3, color="gray", label="Observations", zorder=1)

    ax.plot(band["grid"], band["median"], color="steelblue", linewidth=2, label="Huber trend", zorder=3)
    ax.fill_between(
        band["grid"], band["lower"], band["upper"],
        color="steelblue", alpha=0.2, label=f"{ci}% bootstrap CI", zorder=2,
    )

    ax.set_title("Analyst view: trend estimate with confidence band")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return band

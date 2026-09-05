"""
Publisher-mode visualization: the actual spread of individual values
in the population (median + IQR band) -- NOT a confidence interval on
an estimate. This band does not shrink with more data; it reflects
genuine dispersion (e.g. "half of people this age earn between X and
Y"), which is typically what a general audience reading published
statistics wants to know.

show_points defaults to False deliberately: individual data points
should not be exposed for sensitive data (e.g. salary statistics)
without an explicit, deliberate opt-in.
"""

import numpy as np

from .dispersion import dispersion_by_bin


def plot_publisher_view(x, y, n_bins=10, show_points=False, ax=None, figsize=(10, 6)):
    """
    Plot median + IQR band across quantile bins of x.

    show_points: off by default. Set to True explicitly to overlay
    individual observations -- appropriate for internal analysis, not
    for publishing sensitive data like individual salaries.

    Returns the dispersion_by_bin() result DataFrame for further
    inspection or tabular reporting.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    binned = dispersion_by_bin(x, y, n_bins=n_bins)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.scatter(x, y, s=10, alpha=0.15, color="gray", zorder=1)

    ax.plot(binned["x_center"], binned["median"], color="darkorange", linewidth=2, label="Median", zorder=3)
    ax.fill_between(
        binned["x_center"], binned["q1"], binned["q3"],
        color="darkorange", alpha=0.25, label="Interquartile range (Q1-Q3)", zorder=2,
    )

    ax.set_title("Publisher view: median and population spread")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return binned

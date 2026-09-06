"""
Visualize an already-published quantile trend (e.g. Q1/median/Q3 per
year from an SCB table) over an ordered dimension such as year.

Unlike robustkit.report.dispersion_by_bin (which COMPUTES quantiles
from individual-level data by binning a continuous x), this module
assumes the quantiles are already given -- exactly the case when SCB
publishes Q1/median/Q3 directly, with no underlying individual data
available at all.
"""

import numpy as np
import pandas as pd


def prepare_quantile_trend(df, x_col, q1_col, median_col, q3_col):
    """
    Select and sort the relevant columns from a tidy quantile table
    into a clean, x-sorted DataFrame ready for plotting or reporting.
    """
    trend = df[[x_col, q1_col, median_col, q3_col]].copy()
    trend = trend.rename(columns={q1_col: "q1", median_col: "median", q3_col: "q3"})
    trend = trend.sort_values(x_col).reset_index(drop=True)
    return trend


def plot_quantile_trend(df, x_col, q1_col, median_col, q3_col, show_points=False, ax=None, figsize=(10, 6)):
    """
    Plot an already-published median + IQR band over an ordered x
    (typically year). Visually similar to
    robustkit.report.plot_publisher_view, but the quantiles here are
    taken directly from the data rather than computed by binning --
    appropriate when the quantiles ARE the data (e.g. an SCB table),
    with no individual-level observations to bin in the first place.

    show_points: if True, overlays the three quantile series as points
    in addition to the band -- off by default, matching the same
    "don't imply individual-level data is present" caution as
    plot_publisher_view.

    Returns the prepared, sorted quantile DataFrame.
    """
    import matplotlib.pyplot as plt

    trend = prepare_quantile_trend(df, x_col, q1_col, median_col, q3_col)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(trend[x_col], trend["median"], color="darkorange", linewidth=2, label="Median", zorder=3)
    ax.fill_between(
        trend[x_col], trend["q1"], trend["q3"],
        color="darkorange", alpha=0.25, label="Interquartile range (Q1-Q3)", zorder=2,
    )

    if show_points:
        for col, marker in (("q1", "v"), ("median", "o"), ("q3", "^")):
            ax.scatter(trend[x_col], trend[col], s=20, color="darkorange", marker=marker, zorder=4)

    ax.set_title("Published quantile trend")
    ax.set_xlabel(x_col)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return trend


def quantile_trend_dispersion(df, x_col, q1_col, median_col, q3_col):
    """
    Compute dispersion_ratio ((Q3-Q1)/median) directly from already-
    published quantiles, without needing individual-level data.
    Complements robustkit.report.dispersion_by_bin, which computes the
    same measure FROM individual data by binning -- this version is
    for when the quantiles are already given.
    """
    trend = prepare_quantile_trend(df, x_col, q1_col, median_col, q3_col)
    trend["iqr"] = trend["q3"] - trend["q1"]
    trend["dispersion_ratio"] = np.where(
        trend["median"] != 0, trend["iqr"] / trend["median"], 0.0,
    )
    return trend

"""
Measures of spread that describe actual dispersion within a
population -- distinct from the uncertainty band around a point
estimate (robustkit.core.uncertainty), which shrinks as sample size
grows. The measures here do NOT shrink with more data; they describe
how spread out individual values genuinely are, which is usually what
a general audience reading published statistics wants to know (e.g.
"how much do salaries actually vary at this age?"), as opposed to "how
confident is the analyst in their trend estimate?"
"""

import numpy as np
import pandas as pd


def iqr(y):
    """Interquartile range: Q3 - Q1."""
    y = np.asarray(y, dtype=float)
    q1, q3 = np.percentile(y, [25, 75])
    return float(q3 - q1)


def dispersion_ratio(y):
    """
    (Q3 - Q1) / median -- a scale-free measure of relative spread.
    Rising dispersion_ratio across a trend (e.g. by age) signals
    growing inequality/spread in the population, not growing
    uncertainty in an estimate.
    """
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    if med == 0:
        return 0.0
    return float(iqr(y) / med)


def dispersion_by_bin(x, y, n_bins=10):
    """
    Compute Q1, median, Q3, IQR, and dispersion_ratio of y within each
    of n_bins quantile-based bins of x.

    Returns one row per bin, with x_center as the median x value
    within that bin (used for plotting/interpretation, not the bin
    edges themselves). Bins with fewer than 2 points are dropped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = pd.qcut(x, q=n_bins, duplicates="drop")

    rows = []
    for _, idx in pd.Series(range(len(x))).groupby(bins, observed=True).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        x_bin = x[idx]
        y_bin = y[idx]
        q1, med, q3 = np.percentile(y_bin, [25, 50, 75])
        rows.append({
            "x_center": float(np.median(x_bin)),
            "n": len(idx),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "dispersion_ratio": float((q3 - q1) / med) if med != 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("x_center").reset_index(drop=True)

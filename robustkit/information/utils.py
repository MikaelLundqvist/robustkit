"""
Shared helpers for the information module.
"""

import pandas as pd


def discretize(series, n_bins=5):
    """
    Bin a series into n_bins quantile-based bins if it looks continuous
    (numeric with more distinct values than 2 * n_bins); return it
    as-is (stringified) otherwise.

    Used wherever a feature needs to be treated as a "communicable"
    category -- grouping, stability scoring, minimum group size --
    since those operations require discrete groups.
    """
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins * 2:
        try:
            return pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
        except ValueError:
            return series.astype(str)
    return series.astype(str)

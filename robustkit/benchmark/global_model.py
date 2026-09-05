"""
Compare each segment's observed outcome against what a single global
robust trend would predict, with bootstrap uncertainty on the
difference.

This answers a different question than robustkit.core.stability
(which asks "how does the trend look overall, across fitting
methods?"): here the question is "which groups deviate from the
overall robust trend, once we account for their x values, and how
confident are we in that deviation?"
"""

import numpy as np
import pandas as pd

from ..core.trend import fit_huber_trend, predict_trend
from ..core.uncertainty import bca_bootstrap_ci


def fit_huber_benchmark(x, y, degree=2):
    """
    Fit a single global Huber trend intended to serve as the
    reference/benchmark that segments will be compared against. This
    is just fit_huber_trend under a clearer name for this use case --
    fit it once on the FULL population, not on any one segment.
    """
    return fit_huber_trend(x, y, degree=degree)


def segment_position_report(df, segment_col, x_col, y_col, benchmark_fit=None,
                             degree=2, n_boot=500, ci=95, seed=0):
    """
    For each segment, compare its observed median outcome against what
    the global benchmark trend predicts at each row's x, with a BCa
    bootstrap confidence interval on the (observed - expected)
    difference.

    benchmark_fit: a fit dict from fit_huber_benchmark, ideally fitted
        on the full dataset (df should then be the same full dataset
        this was fitted on, not a pre-filtered subset -- otherwise the
        benchmark isn't a genuine "overall" reference for the segments
        being compared). If omitted, one is fitted here on all of df.

    Returns one row per segment: n, observed_median, expected_median,
    difference (bootstrap point estimate), and the CI bounds.
    """
    if benchmark_fit is None:
        benchmark_fit = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )

    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        expected = predict_trend(benchmark_fit, x)

        def difference_stat(x_, y_, _fit=benchmark_fit):
            exp = predict_trend(_fit, x_)
            return float(np.median(y_ - exp))

        ci_result = bca_bootstrap_ci(x, y, difference_stat, n_boot=n_boot, ci=ci, seed=seed)

        rows.append({
            "segment": segment_value,
            "n": len(group),
            "observed_median": float(np.median(y)),
            "expected_median": float(np.median(expected)),
            "difference": ci_result["estimate"],
            "ci_lower": ci_result["lower"],
            "ci_upper": ci_result["upper"],
        })

    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)

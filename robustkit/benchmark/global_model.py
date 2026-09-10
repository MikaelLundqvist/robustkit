"""
Compare each segment's observed outcome against what a benchmark model
predicts, with bootstrap uncertainty on the difference.

This answers a different question than robustkit.core.stability
(which asks "how does the trend look overall, across fitting
methods?"): here the question is "which groups deviate from the
benchmark, once we account for whatever the benchmark model
considers, and how confident are we in that deviation?"

Two ways to supply a benchmark:

    1. Default: a single global Huber trend on one continuous x,
       fitted automatically (fit_huber_benchmark). This is the
       original, simplest case.
    2. Custom: any object exposing predict(dataframe) -> array, e.g. a
       richer model with several predictors (age, level, overtime
       status, cluster, ...). segment_position_report only needs the
       model to expose predict(); it never inspects what the model
       actually uses internally.

Small segments (fewer than MIN_POINTS_FOR_CI observations) still get
observed/expected/difference reported, but no bootstrap confidence
interval -- BCa's jackknife step in particular breaks down (or becomes
statistically meaningless) at very small n, so it is skipped rather
than attempted and silently trusted.
"""

import numpy as np
import pandas as pd

from ..core.trend import fit_huber_trend, predict_trend
from ..core.uncertainty import bca_bootstrap_ci_by_index

MIN_POINTS_FOR_CI = 20


def fit_huber_benchmark(x, y, degree=2):
    """
    Fit a single global Huber trend intended to serve as the default
    benchmark that segments will be compared against, when no custom
    benchmark model is supplied. Fit it once on the FULL population,
    not on any one segment.
    """
    return fit_huber_trend(x, y, degree=degree)


def benchmark_predict(benchmark_fit, data, x_col=None):
    """
    Generic prediction wrapper, supporting two kinds of benchmark:

      1. A robustkit trend fit dict (from fit_huber_trend /
         fit_huber_benchmark): requires x_col, predicts from that
         single column of `data`.
      2. Any custom object exposing predict(dataframe) -> array: gets
         the full `data` (a DataFrame, or a row-subset of one) and is
         responsible for extracting whatever columns it needs itself.
         This is what makes segment_position_report model-agnostic --
         it never needs to know whether the model uses one column or
         several.
    """
    if isinstance(benchmark_fit, dict):
        if x_col is None:
            raise ValueError(
                "x_col must be supplied when using a robustkit trend-fit "
                "dictionary as the benchmark (it predicts from a single "
                "column). Custom models exposing predict(dataframe) do "
                "not need x_col."
            )
        return predict_trend(benchmark_fit, data[x_col].to_numpy(dtype=float))

    if hasattr(benchmark_fit, "predict"):
        return benchmark_fit.predict(data)

    raise TypeError(
        "benchmark_fit must be either a robustkit trend-fit dictionary "
        "or an object exposing predict(dataframe)."
    )


def segment_position_report(df, segment_col, y_col, benchmark_fit=None, x_col=None,
                             degree=2, n_boot="auto", ci=95, seed=0):
    """
    Compare each segment's outcome against a benchmark model.

    Default (single-column) benchmark:

        segment_position_report(df, segment_col="segment", x_col="age", y_col="salary")

    Custom (multi-column) benchmark:

        segment_position_report(df, segment_col="segment", y_col="salary", benchmark_fit=my_model)

    where my_model exposes predict(dataframe) -> array-like, and can
    use as many columns of `df` internally as it needs (age, level,
    overtime status, cluster, ...) -- segment_position_report doesn't
    need to know.

    Segments with fewer than MIN_POINTS_FOR_CI (default 20)
    observations still get observed_median / expected_median /
    difference, but ci_lower / ci_upper are NaN and ci_available is
    False -- a BCa confidence interval (which relies on a jackknife
    step) is not attempted for populations that small.

    For segments at or above the threshold, the confidence interval on
    the difference is a full BCa (bias-corrected and accelerated)
    bootstrap interval, via bca_bootstrap_ci_by_index -- not a plain
    percentile bootstrap.
    """
    if benchmark_fit is None:
        if x_col is None:
            raise ValueError("x_col must be supplied when no custom benchmark model is provided.")
        benchmark_fit = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )

    rows = []

    for segment_value, group in df.groupby(segment_col, observed=True):
        group = group.reset_index(drop=True)
        n = len(group)

        y = group[y_col].to_numpy(dtype=float)
        expected = benchmark_predict(benchmark_fit, group, x_col=x_col)

        observed_median = float(np.median(y))
        expected_median = float(np.median(expected))
        difference = float(np.median(y - expected))

        if n < MIN_POINTS_FOR_CI:
            rows.append({
                "segment": segment_value,
                "n": n,
                "observed_median": observed_median,
                "expected_median": expected_median,
                "difference": difference,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "ci_available": False,
            })
            continue

        def stat_by_index(idx, _group=group, _fit=benchmark_fit):
            sub = _group.iloc[idx]
            y_sub = sub[y_col].to_numpy(dtype=float)
            exp_sub = benchmark_predict(_fit, sub, x_col=x_col)
            return float(np.median(y_sub - exp_sub))

        ci_result = bca_bootstrap_ci_by_index(n, stat_by_index, n_boot=n_boot, ci=ci, seed=seed)

        rows.append({
            "segment": segment_value,
            "n": n,
            "observed_median": observed_median,
            "expected_median": expected_median,
            "difference": ci_result["estimate"],
            "ci_lower": ci_result["lower"],
            "ci_upper": ci_result["upper"],
            "ci_available": True,
        })

    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)

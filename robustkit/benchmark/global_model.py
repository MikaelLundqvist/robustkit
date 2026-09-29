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
                             degree=2, n_boot="auto", ci=95, seed=0, jackknife_cap=1000):
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

    jackknife_cap: see bca_bootstrap_ci_by_index's docstring -- caps
    the O(n) jackknife pass at a fixed sample size for large segments
    (1000 by default). Pass None to always use the full jackknife.
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
        expected = np.asarray(benchmark_predict(benchmark_fit, group, x_col=x_col), dtype=float)

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

        # Precompute the per-row residual ONCE -- it doesn't change
        # between resamples, since resampling only changes WHICH rows
        # are included, not the model's prediction for a given
        # original row. See segment_awareness.reports for the real,
        # measured impact of this on large segments (minutes -> ~3s).
        residual = y - expected

        def stat_by_index(idx, _residual=residual):
            return float(np.median(_residual[idx]))

        ci_result = bca_bootstrap_ci_by_index(n, stat_by_index, n_boot=n_boot, ci=ci, seed=seed, jackknife_cap=jackknife_cap)

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


def plot_segment_vs_benchmark(df, segment_col, segment_values, y_col, x_col, benchmark_fit=None,
                               degree=2, frac=0.3, figsize=(10, 6), ax=None, colors=None):
    """
    Visualize WHERE, across x_col, one or more segments diverge from a
    benchmark -- the visual counterpart to segment_position_report,
    which only reports a single summary difference per segment.

    For each segment in segment_values, plots:
      - the segment's actual (x_col, y_col) observations, as points
      - the segment's own OBSERVED trend, via LOWESS (a flexible,
        assumption-free local smoother -- deliberately not a Huber/
        Tukey/OLS fit at a fixed polynomial degree, since the point
        here is to see the segment's raw shape without imposing one)
      - the BENCHMARK's expected trend for that segment, as a dashed
        line of the same color

    A single summary difference (as segment_position_report reports)
    can arise from several different underlying patterns that look
    identical in a table but mean very different things: the whole
    segment sitting a bit below the benchmark throughout, only its
    younger members sitting far below, only its older members, or two
    opposing effects that happen to cancel out in the median. This
    plot makes those patterns visible; the table alone cannot
    distinguish them.

    benchmark_fit: as in segment_position_report -- a robustkit
    trend-fit dict (requires x_col) or any object exposing
    predict(dataframe). If not supplied, a Huber benchmark is fit once
    across the full population.

    frac: the LOWESS smoothing fraction (passed straight through to
    statsmodels) -- the proportion of points used to estimate each
    local value. Smaller values follow the data more closely (and can
    look noisy); larger values smooth more aggressively. Requires
    statsmodels (imported lazily here, same as fit_tukey_trend).

    This plot is diagnostic, not a source of numbers for further
    analysis -- LOWESS here is used only to visualize a segment's
    observed shape, not to compute an estimate other functions build
    on, which is why it's used here specifically rather than
    elsewhere in the package.

    Returns the matplotlib Axes.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import matplotlib.pyplot as plt

    benchmark_fit_resolved = benchmark_fit
    if benchmark_fit_resolved is None:
        benchmark_fit_resolved = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or default_colors

    for i, seg_value in enumerate(segment_values):
        color = colors[i % len(colors)]
        sub = df[df[segment_col] == seg_value]
        if len(sub) == 0:
            continue

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        ax.scatter(x, y, s=12, alpha=0.35, color=color, zorder=2)

        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2,
                label=f"{seg_value} (observed)", zorder=3)

        grid = np.linspace(x.min(), x.max(), 100)
        grid_df = pd.DataFrame({x_col: grid})
        for col in df.columns:
            if col not in (x_col, y_col) and sub[col].nunique() == 1:
                grid_df[col] = sub[col].iloc[0]
        expected = np.asarray(benchmark_predict(benchmark_fit_resolved, grid_df, x_col=x_col), dtype=float)
        ax.plot(grid, expected, color=color, linewidth=2, linestyle="--",
                label=f"{seg_value} (benchmark)", zorder=3)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Observed vs. benchmark trend by {segment_col}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def benchmark_goodness_of_fit(df, y_col, benchmark_fit, x_col=None):
    """
    Evaluate an ALREADY-FITTED benchmark's overall explanatory power --
    r_squared, rmse, mae -- against the actual values in y_col.

    This answers a different question than benchmark_screening_report:
    screening asks "is there specific structure (a polynomial degree,
    a categorical variable, an interaction) that seems to be missing?"
    Even a screen that turns up nothing -- every candidate's residual
    mutual information near zero -- says nothing about how much the
    benchmark explains overall; a benchmark can be "complete", in the
    sense that no obvious structure is left to add, while still
    explaining relatively little of y_col's variance if the underlying
    noise is simply large. This function answers that separate
    question directly.

    Unlike core.goodness_of_fit.goodness_of_fit, which FITS its own
    trend internally (via fit_huber_trend/fit_tukey_trend/
    fit_ols_trend at a chosen degree) and therefore only evaluates a
    single-column robustkit trend, this function evaluates a benchmark
    that's already been fit elsewhere -- a robustkit trend-fit dict, or
    any custom, potentially multivariate object exposing
    predict(dataframe), exactly like benchmark_predict elsewhere in
    this module. It never re-fits anything itself.

    Returns a dict: {"r_squared", "rmse", "mae"}, computed the same
    way as goodness_of_fit for direct comparability.
    """
    y = df[y_col].to_numpy(dtype=float)
    expected = np.asarray(benchmark_predict(benchmark_fit, df, x_col=x_col), dtype=float)
    residuals = y - expected
    var_y = np.var(y)
    r_squared = 1 - np.var(residuals) / var_y if var_y > 0 else 0.0

    return {
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def benchmark_screening_report(df, y_col, benchmark_fit, x_col=None, candidate_cols=None,
                                degrees=(1, 2, 3, 4), interaction_pairs=None, degree=2, n_bins=5, seed=0):
    """
    Screen a fitted benchmark for likely-missing structure, without
    building or fitting any new benchmark model automatically. Two
    genuinely different questions, answered with two genuinely
    different tools -- NOT one unified ranking, because mutual
    information cannot distinguish them (see below):

    1. "Does x_col's current polynomial degree capture its true
       shape?" Answered by re-running compare_polynomial_degrees on
       the RAW (x_col, y_col) relationship (Chapter 5's tool),
       independent of what degree the benchmark actually used --
       compare the r_squared_gain at degrees beyond your benchmark's
       own degree to judge whether more curvature is worth adding.

       Mutual information is NOT used for this, on purpose: on a
       domain where x_col is always positive (e.g. age in a typical
       range), x_col, x_col**2, and x_col**3 are all monotonic
       (invertible) transforms of each other, and mutual information
       is invariant to invertible reparametrization -- so MI(x_col,
       residual), MI(x_col**2, residual), and MI(x_col**3, residual)
       come out nearly identical REGARDLESS of which degree actually
       fits best. MI can detect THAT a dependency exists; it cannot
       tell you WHAT SHAPE that dependency has. Verified directly:
       feeding a residual from a deliberately under-fit linear
       benchmark (true relationship quadratic) into an MI-based
       ranking gave age/age^2/age^3 mutual information within about
       2% of each other -- indistinguishable, despite the benchmark
       clearly needing degree 2, not 1.

    2. "Is there a categorical variable, or a combination of two, that
       still explains part of the residual?" Answered by fitting the
       benchmark's residual (y_col minus benchmark_fit's prediction)
       as the TARGET of Chapter 8's rank_features/information_efficiency
       machinery -- unlike polynomial degree, a genuinely-missing
       categorical variable (or interaction) is not a reparametrization
       of anything already in the model, so mutual information detects
       it directly and reliably. Verified directly: residuals from a
       benchmark missing two real categorical drivers showed 0.87-1.22
       bits of mutual information with them; residuals from the
       complete benchmark showed exactly 0.0 bits with the same
       columns.

    candidate_cols: categorical columns to check individually against
        the residual (defaults to every non-numeric column in df other
        than y_col and x_col).

    interaction_pairs: list of (col_a, col_b) tuples to check as
        combined categories (e.g. [("JobFamily", "level")] checks
        whether the specific JobFamily-level COMBINATION explains
        residual variance beyond what either column explains alone).
        Defaults to every pairwise combination of candidate_cols if
        not supplied -- convenient, but grows combinatorially and
        raises real multiple-testing risk (some pairs will show
        nonzero efficiency from chance alone, especially with many
        candidates or a modest sample size). Treat every entry in the
        `interactions` result as "worth investigating further", not as
        a confirmed missing term -- this function screens for
        candidates, it does not validate them.

    degree: the polynomial degree used to fit RESIDUALS in step 2
        internally (irrelevant to step 1, which always tests the
        requested `degrees` directly on the raw relationship).

    Returns a dict with three entries:
      "fit": benchmark_goodness_of_fit's {"r_squared", "rmse", "mae"}
          for the benchmark AS GIVEN -- how much it explains overall,
          answering a different question than the two screens below
          (see benchmark_goodness_of_fit's docstring for why this
          isn't redundant with an empty screening result).
      "polynomial": compare_polynomial_degrees(x_col, y_col, degrees) --
          only present if x_col is supplied.
      "interactions": one row per candidate_col and per interaction
          pair, each with mutual_information, entropy_bits, and
          information_efficiency against the benchmark's residual,
          sorted by information_efficiency descending -- ranking by
          efficiency rather than raw mutual_information specifically
          to avoid favoring a high-cardinality candidate (many
          combined categories) just because it has more complexity to
          spend, the same trap Chapter 8's zip_code example
          demonstrated.
    """
    from ..core.goodness_of_fit import compare_polynomial_degrees
    from ..information.mutual_info import rank_features

    result = {"fit": benchmark_goodness_of_fit(df, y_col=y_col, benchmark_fit=benchmark_fit, x_col=x_col)}

    if x_col is not None:
        result["polynomial"] = compare_polynomial_degrees(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degrees=degrees,
        )

    expected = np.asarray(benchmark_predict(benchmark_fit, df, x_col=x_col), dtype=float)
    residual = df[y_col].to_numpy(dtype=float) - expected

    if candidate_cols is None:
        exclude = {y_col} | ({x_col} if x_col is not None else set())
        candidate_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]

    if interaction_pairs is None:
        interaction_pairs = [
            (candidate_cols[i], candidate_cols[j])
            for i in range(len(candidate_cols))
            for j in range(i + 1, len(candidate_cols))
        ]

    screen_df = pd.DataFrame({"_residual": residual})
    for col in candidate_cols:
        screen_df[col] = df[col].to_numpy()
    for col_a, col_b in interaction_pairs:
        combo_name = f"{col_a} x {col_b}"
        screen_df[combo_name] = df[col_a].astype(str) + " / " + df[col_b].astype(str)

    ranking = rank_features(screen_df, target="_residual", seed=seed, n_bins=n_bins)
    result["interactions"] = ranking.sort_values("information_efficiency", ascending=False).reset_index(drop=True)

    return result

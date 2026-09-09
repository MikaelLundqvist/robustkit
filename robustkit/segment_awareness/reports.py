"""
Automatic hierarchical segmentation + analysis, in one call.

Both functions here take an ordered list of grouping columns
(segment_cols) and build the same kind of fallback hierarchy
hierarchical_segment already supports -- falling back from most
specific to least specific by dropping columns from the FRONT of the
list first:

    [JobFamily, Level, OvertimeStatus]
        -> [Level, OvertimeStatus]
        -> [OvertimeStatus]
        -> "ALL" (hierarchical_segment's own catch-all)

This ordering matches the intuition that the columns are listed most-
specific-first: the analysis prefers the finest grouping it can
support at min_size, falling back toward broader categories rather
than collapsing straight to a single population.
"""

import numpy as np
import pandas as pd

from ..segmentation.hierarchy import hierarchical_segment
from ..segmentation.apply import apply_by_segment
from ..core.stability import model_stability_pct
from ..benchmark.global_model import segment_position_report


def _build_hierarchy(segment_cols):
    """[A, B, C] -> [[A, B, C], [B, C], [C]] -- drop from the front."""
    return [segment_cols[i:] for i in range(len(segment_cols))]


def _attach_segment_level(report, segmented):
    """Add a `segment_level` column to a report, indicating which tier
    of the hierarchy each reported segment was actually assigned to
    (0 = finest -- see hierarchical_segment's docstring)."""
    level_map = segmented.groupby("segment_id")["segment_level"].first()
    report = report.copy()
    report["segment_level"] = report["segment"].map(level_map)
    return report


def segment_stability_report(df, x_col, y_col, segment_cols, min_size=20, min_points=5, degree=2, **stability_kwargs):
    """
    Build a hierarchical segmentation from segment_cols automatically,
    then run model_stability_pct independently within each resulting
    segment.

    Equivalent to calling hierarchical_segment(df, hierarchy, min_size)
    followed by apply_by_segment(..., model_stability_pct) by hand --
    this just builds the hierarchy list for you from a flat,
    most-specific-first column list.

    Returns the same report shape as apply_by_segment, plus a
    `segment_level` column showing which tier of the hierarchy each
    segment was actually assigned to.
    """
    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    report = apply_by_segment(
        segmented, segment_col="segment_id", x_col=x_col, y_col=y_col,
        analysis_fn=model_stability_pct, min_points=min_points, degree=degree, **stability_kwargs,
    )
    return _attach_segment_level(report, segmented)


def segment_benchmark_report(df, y_col, segment_cols, benchmark_fit=None, x_col=None,
                              degree=2, min_size=20, n_boot=500, ci=95, seed=0):
    """
    Build a hierarchical segmentation from segment_cols automatically,
    then run segment_position_report within the resulting segments.

    Accepts the same benchmark_fit / x_col contract as
    segment_position_report (default single-column Huber trend, or any
    custom model exposing predict(dataframe)).

    Returns the same report shape as segment_position_report, plus a
    `segment_level` column showing which tier of the hierarchy each
    segment was actually assigned to.
    """
    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    report = segment_position_report(
        segmented, segment_col="segment_id", y_col=y_col, benchmark_fit=benchmark_fit, x_col=x_col,
        degree=degree, n_boot=n_boot, ci=ci, seed=seed,
    )
    return _attach_segment_level(report, segmented)


def mad_outlier_report(df, y_col, segment_cols, benchmark_fit=None, x_col=None, degree=2,
                        min_size=20, k=3.0, direction="negative", id_cols=None):
    """
    Flag individuals whose residual from a benchmark model is an
    outlier relative to their OWN segment's typical spread (MAD),
    rather than the population as a whole -- comparing someone against
    an irrelevant reference population is exactly the failure mode
    hierarchical segmentation exists to avoid.

    Segments are built automatically and hierarchically from
    segment_cols (most-specific-first, falling back to broader
    groupings when a segment is smaller than min_size -- see
    hierarchical_segment), so an individual in a small segment still
    gets compared against a sensibly-sized reference population,
    rather than being skipped or compared against an unrelated one.

    Flagging logic: within each individual's assigned segment, compute
    the segment's median residual and MAD (median absolute deviation,
    scaled to be comparable to a standard deviation under normality).
    An individual is flagged if their residual is more than k MADs
    below (direction="negative"), above (direction="positive"), or
    either side of (direction="two_sided") their segment's OWN median
    residual -- not literally "residual < -k*MAD" against zero, since
    centering on the segment's median residual avoids flagging an
    entire segment just because the benchmark is systematically biased
    for that segment as a whole.

    k: number of MADs from the segment's median residual beyond which
        a point is flagged. 3.0 is a conventional threshold; some
        sources use 3.5 for a stricter criterion.
    direction: "negative" (default -- flag underperformance relative
        to the benchmark, the typical HR/union use case), "positive",
        or "two_sided".
    id_cols: columns identifying each individual in the output (e.g.
        an employee ID). Falls back to the DataFrame's index as
        "row_id" if omitted, matching negative_deviation_report's
        convention.

    Segments where MAD is exactly 0 (a degenerate segment where every
    residual is identical -- most often a very small segment) cannot
    be meaningfully flagged against and are treated as having an
    infinite threshold, rather than flagging everything in them or
    raising a divide-by-zero error.

    Returns one row per individual: id_cols (or row_id), actual,
    expected, residual, residual_pct, segment, segment_level,
    segment_mad, threshold, flagged.
    """
    if direction not in ("negative", "positive", "two_sided"):
        raise ValueError(f"direction must be 'negative', 'positive', or 'two_sided', got {direction!r}")

    benchmark_fit_resolved = benchmark_fit
    if benchmark_fit_resolved is None:
        if x_col is None:
            raise ValueError("x_col must be supplied when no custom benchmark model is provided.")
        from ..benchmark.global_model import fit_huber_benchmark
        benchmark_fit_resolved = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )

    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    from ..benchmark.global_model import benchmark_predict

    y = segmented[y_col].to_numpy(dtype=float)
    expected = np.asarray(benchmark_predict(benchmark_fit_resolved, segmented, x_col=x_col), dtype=float)
    residual = y - expected
    residual_pct = np.where(expected != 0, residual / expected * 100, np.nan)

    segmented = segmented.copy()
    segmented["_residual"] = residual

    segment_median_residual = segmented.groupby("segment_id")["_residual"].median()
    segment_mad = segmented.groupby("segment_id")["_residual"].apply(
        lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))
    )

    median_per_row = segmented["segment_id"].map(segment_median_residual).to_numpy(dtype=float)
    mad_per_row = segmented["segment_id"].map(segment_mad).to_numpy(dtype=float)

    safe_mad = np.where(mad_per_row > 0, mad_per_row, np.inf)
    z = (residual - median_per_row) / safe_mad
    threshold = k * mad_per_row

    if direction == "negative":
        flagged = z < -k
    elif direction == "positive":
        flagged = z > k
    else:  # two_sided
        flagged = np.abs(z) > k

    if id_cols:
        result = segmented[id_cols].reset_index(drop=True)
    else:
        result = pd.DataFrame({"row_id": segmented.index})

    result["actual"] = y
    result["expected"] = expected
    result["residual"] = residual
    result["residual_pct"] = residual_pct
    result["segment"] = segmented["segment_id"].to_numpy()
    result["segment_level"] = segmented["segment_level"].to_numpy()
    result["segment_mad"] = mad_per_row
    result["threshold"] = threshold
    result["flagged"] = flagged

    return result


def export_outlier_pdf(df, y_col, segment_cols, x_col, path, benchmark_fit=None, degree=2,
                        min_size=20, k=3.0, direction="negative", id_cols=None,
                        min_points_to_plot=5, figsize=(10, 6)):
    """
    Render one chart per segment -- built from the same hierarchical
    segmentation and MAD-outlier flagging as mad_outlier_report --
    showing every observation in that segment, the benchmark's
    expected values, and flagged outliers marked distinctly. One page
    per segment in a single PDF.

    Purpose: visual verification, not primary analysis.
    mad_outlier_report gives a numeric answer ("this person is 3.5
    MADs below trend"); this gives a visual one you can look at and
    confirm "yes, they really are far below the curve" -- two
    independent checks that, when they agree, increase confidence in
    the result more than either alone.

    The plotted "expected" line uses the EXACT same values that were
    used to compute residuals and flag outliers (via benchmark_predict
    on the actual benchmark_fit, whether that's the default Huber
    trend or a custom multi-column model) -- not a separately re-fit
    visual approximation. If a custom model depends on more than
    x_col, connecting its expected values (sorted by x_col) can look
    less smooth than a pure Huber curve; that's expected and itself
    informative, not a rendering bug.

    x_col is required here (unlike mad_outlier_report's optional
    x_col with a custom benchmark_fit), since every page needs
    something to put on the x-axis for a 2D chart.

    Segments with fewer than min_points_to_plot observations are
    skipped in the PDF (a chart with a handful of points isn't
    meaningfully verifiable) but still appear in the numeric
    mad_outlier_report -- call that separately if you need every
    segment's numbers regardless of plot size.

    Returns `path`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    report = mad_outlier_report(
        df, y_col=y_col, segment_cols=segment_cols, benchmark_fit=benchmark_fit, x_col=x_col,
        degree=degree, min_size=min_size, k=k, direction=direction, id_cols=id_cols,
    )

    # mad_outlier_report doesn't return x_col itself; recompute the
    # identical (deterministic) segmentation to attach it positionally.
    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    report = report.copy()
    report["_x"] = segmented[x_col].to_numpy(dtype=float)

    segments_in_order = sorted(str(s) for s in report["segment"].dropna().unique())

    with PdfPages(path) as pdf:
        for segment_name in segments_in_order:
            seg_data = report[report["segment"].astype(str) == segment_name]
            if len(seg_data) < min_points_to_plot:
                continue

            fig, ax = plt.subplots(figsize=figsize)

            order = seg_data["_x"].to_numpy().argsort()
            x_sorted = seg_data["_x"].to_numpy()[order]
            expected_sorted = seg_data["expected"].to_numpy()[order]

            not_flagged = seg_data[~seg_data["flagged"]]
            flagged = seg_data[seg_data["flagged"]]

            ax.scatter(
                not_flagged["_x"], not_flagged["actual"],
                s=15, alpha=0.5, color="steelblue", label="Observations", zorder=2,
            )
            ax.scatter(
                flagged["_x"], flagged["actual"],
                s=70, color="red", marker="x", linewidths=2,
                label=f"Flagged outliers (n={len(flagged)})", zorder=4,
            )
            ax.plot(x_sorted, expected_sorted, color="darkorange", linewidth=2, label="Benchmark (expected)", zorder=3)

            ax.set_title(f"{segment_name}  (n={len(seg_data)}, {len(flagged)} flagged, k={k})")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    return path

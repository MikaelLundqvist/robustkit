"""
Automatic hierarchical segmentation + analysis, in one call.

Two complementary families of functions live here, answering two
different questions:

  EXCLUSIVE (segment_stability_report, segment_benchmark_report,
  mad_outlier_report): each individual is assigned to exactly ONE
  segment -- their most specific grouping that meets min_size, falling
  back to a broader one otherwise. Answers "what is the single most
  relevant reference population for THIS individual?"

  DRILLDOWN (segment_benchmark_drilldown_report,
  mad_outlier_drilldown_report): every level of the hierarchy is
  reported independently and WITHOUT exclusive assignment -- the same
  individual can appear in multiple rows (e.g. once in a
  JobFamily x Level x OT row, and again in the broader Level x OT
  row), whenever both groupings independently meet min_size. Answers
  "what does every granularity level look like on its own?"

Both families build the same kind of fallback hierarchy from a flat,
most-specific-first list of columns -- falling back from most specific
to least specific by dropping columns from the FRONT of the list
first:

    [JobFamily, Level, OvertimeStatus]
        -> [Level, OvertimeStatus]
        -> [OvertimeStatus]
        -> "ALL" (hierarchical_segment's own catch-all -- EXCLUSIVE
                  family only; drilldown has no catch-all, since it
                  doesn't need one)

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
from ..core.diagnostics import cooks_diagnostic, cook_impact
from ..core.trend import fit_huber_trend, predict_trend
from ..core.uncertainty import bca_bootstrap_ci_by_index
from ..report.visualize_huber_iqr import plot_huber_iqr
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
                              degree=2, min_size=20, n_boot="auto", ci=95, seed=0):
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
        "row_id" if omitted, matching deviation_report's
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
                        min_points_to_plot=5, annotate_flagged_with=None, mode="benchmark",
                        title_fn=None, figsize=(10, 6)):
    """
    Render one chart per segment -- built from the same hierarchical
    segmentation and MAD-outlier flagging as mad_outlier_report --
    showing every observation in that segment, a reference trend, and
    flagged outliers marked distinctly. One page per segment in a
    single PDF.

    Purpose: visual verification, not primary analysis.
    mad_outlier_report gives a numeric answer ("this person is 3.5
    MADs below trend"); this gives a visual one you can look at and
    confirm "yes, they really are far below the curve" -- two
    independent checks that, when they agree, increase confidence in
    the result more than either alone.

    mode: "benchmark" (default) plots the benchmark's expected values
        -- the EXACT values used to compute residuals and flag
        outliers, connected in x-order. Most consistent with the
        numbers in mad_outlier_report, since the flagging always uses
        benchmark residuals regardless of this setting.
        "local" instead fits a fresh Huber trend WITHIN each segment
        on (x_col, y_col) alone, and plots that smooth curve --
        typically easier to read for "where does this person sit
        relative to their immediate colleagues?" Note that flagging
        still uses the BENCHMARK residuals even in "local" mode, so a
        point can visually sit close to the local curve yet still be
        flagged (or vice versa) if the benchmark disagrees with the
        segment's own local trend -- surfacing that gap is often
        useful information, not a rendering inconsistency.

    Each page includes an info box (n, flagged count/percentage, the
    segment's MAD, and the flagging rule) so a reviewer can understand
    "why are these points flagged?" directly from the chart, without
    needing to cross-reference the numeric report.

    annotate_flagged_with: optional column name (present in id_cols)
        to label each flagged point with directly on the chart -- e.g.
        an employee ID -- for quick cross-referencing against the
        numeric report.

    title_fn: optional callable(segment_name, info) -> str, where info
        is a dict with {"n", "n_flagged", "pct_flagged", "k",
        "segment_mad"}. Lets callers apply their own domain-specific
        segment naming (e.g. translating internal codes into a
        readable label) without robustkit needing to know anything
        about that domain. Defaults to a generic title if not given.

    The plotted "expected" line (mode="benchmark") uses the EXACT same
    values that were used to compute residuals and flag outliers (via
    benchmark_predict on the actual benchmark_fit, whether that's the
    default Huber trend or a custom multi-column model) -- not a
    separately re-fit visual approximation. If a custom model depends
    on more than x_col, connecting its expected values (sorted by
    x_col) can look less smooth than a pure Huber curve; that's
    expected and itself informative, not a rendering bug.

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
    if mode not in ("benchmark", "local"):
        raise ValueError(f"mode must be 'benchmark' or 'local', got {mode!r}")

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

    if direction == "negative":
        rule_text = f"residual < -{k:.1f} \u00d7 MAD"
    elif direction == "positive":
        rule_text = f"residual > {k:.1f} \u00d7 MAD"
    else:
        rule_text = f"|residual| > {k:.1f} \u00d7 MAD"

    with PdfPages(path) as pdf:
        for segment_name in segments_in_order:
            seg_data = report[report["segment"].astype(str) == segment_name]
            if len(seg_data) < min_points_to_plot:
                continue

            fig, ax = plt.subplots(figsize=figsize)

            if mode == "benchmark":
                order = seg_data["_x"].to_numpy().argsort()
                curve_x = seg_data["_x"].to_numpy()[order]
                curve_y = seg_data["expected"].to_numpy()[order]
                curve_label = "Benchmark (expected)"
            else:
                from ..core.trend import fit_huber_trend, predict_trend
                seg_x = seg_data["_x"].to_numpy(dtype=float)
                seg_y = seg_data["actual"].to_numpy(dtype=float)
                local_fit = fit_huber_trend(seg_x, seg_y, degree=degree)
                curve_x = np.linspace(seg_x.min(), seg_x.max(), 200)
                curve_y = predict_trend(local_fit, curve_x)
                curve_label = "Local Huber trend (segment)"

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
            ax.plot(curve_x, curve_y, color="darkorange", linewidth=2, label=curve_label, zorder=3)

            if annotate_flagged_with and annotate_flagged_with in flagged.columns:
                for _, row in flagged.iterrows():
                    ax.annotate(
                        str(row[annotate_flagged_with]), (row["_x"], row["actual"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points", zorder=5,
                    )

            n_total = len(seg_data)
            n_flagged = len(flagged)
            pct_flagged = (n_flagged / n_total * 100) if n_total else 0.0
            segment_mad = float(seg_data["segment_mad"].iloc[0]) if "segment_mad" in seg_data.columns and n_total else float("nan")

            info_text = (
                f"n = {n_total}\n"
                f"Flagged = {n_flagged} ({pct_flagged:.1f}%)\n"
                f"MAD = {segment_mad:,.0f}\n"
                f"Rule: {rule_text}"
            )
            ax.text(
                0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                horizontalalignment="right", verticalalignment="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85}, zorder=6,
            )

            if title_fn is not None:
                title_info = {
                    "n": n_total, "n_flagged": n_flagged, "pct_flagged": pct_flagged,
                    "k": k, "segment_mad": segment_mad,
                }
                title = title_fn(segment_name, title_info)
            else:
                title = f"{segment_name}  (n={n_total}, {n_flagged} flagged, k={k})"
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# DRILLDOWN family: every hierarchy level reported independently, no
# exclusive assignment. See module docstring for how this differs from
# the EXCLUSIVE family above.
# ---------------------------------------------------------------------------

def _label_group_name(name):
    """groupby() gives a scalar for a single column, a tuple for
    multiple -- normalize both to a single underscore-joined string."""
    if isinstance(name, tuple):
        return "_".join(str(x) for x in name)
    return str(name)


def segment_benchmark_drilldown_report(df, y_col, segment_cols, benchmark_fit=None, x_col=None,
                                        degree=2, min_size=20, n_boot="auto", ci=95, seed=0):
    """
    Report segment_position_report-style results at EVERY level of the
    hierarchy simultaneously, without exclusive assignment.

    The same individual can appear in multiple rows here -- once in a
    JobFamily x Level x OT row, and again in the broader Level x OT
    row, if both groupings independently meet min_size. This answers
    "what does every granularity level look like on its own?" For the
    complementary question -- "what is the single most relevant
    reference population for THIS individual?" -- use
    segment_benchmark_report instead.

    Returns one row per (qualifying group, hierarchy level), with a
    segment_level column (0 = finest) disambiguating rows that might
    otherwise share a segment label across levels.
    """
    if benchmark_fit is None:
        if x_col is None:
            raise ValueError("x_col must be supplied when no custom benchmark model is provided.")
        from ..benchmark.global_model import fit_huber_benchmark
        benchmark_fit = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )

    from ..benchmark.global_model import benchmark_predict

    hierarchy = _build_hierarchy(list(segment_cols))
    rows = []

    for level, cols in enumerate(hierarchy):
        for name, sub in df.groupby(cols, observed=True):
            if len(sub) < min_size:
                continue

            label = _label_group_name(name)
            y = sub[y_col].to_numpy(dtype=float)
            expected = np.asarray(benchmark_predict(benchmark_fit, sub, x_col=x_col), dtype=float)
            n = len(sub)

            def stat_by_index(idx, _sub=sub, _fit=benchmark_fit):
                s = _sub.iloc[idx]
                y_s = s[y_col].to_numpy(dtype=float)
                exp_s = benchmark_predict(_fit, s, x_col=x_col)
                return float(np.median(y_s - exp_s))

            ci_result = bca_bootstrap_ci_by_index(n, stat_by_index, n_boot=n_boot, ci=ci, seed=seed)

            rows.append({
                "segment": label,
                "segment_level": level,
                "n": n,
                "observed_median": float(np.median(y)),
                "expected_median": float(np.median(expected)),
                "difference": ci_result["estimate"],
                "ci_lower": ci_result["lower"],
                "ci_upper": ci_result["upper"],
            })

    return pd.DataFrame(rows).sort_values(["segment_level", "segment"]).reset_index(drop=True)


def mad_outlier_drilldown_report(df, y_col, segment_cols, benchmark_fit=None, x_col=None, degree=2,
                                  min_size=20, k=3.0, direction="negative", id_cols=None):
    """
    Flag MAD outliers at EVERY level of the hierarchy independently,
    without exclusive assignment -- the same individual can appear
    (and be flagged, or not) multiple times across levels, once per
    qualifying grouping they belong to.

    Complementary to mad_outlier_report, which assigns each individual
    to exactly one (their most specific) segment. See the module
    docstring for when to use which.

    Same flagging logic, k, and direction semantics as
    mad_outlier_report -- see that function's docstring for details.
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

    from ..benchmark.global_model import benchmark_predict

    hierarchy = _build_hierarchy(list(segment_cols))
    blocks = []

    for level, cols in enumerate(hierarchy):
        for name, sub in df.groupby(cols, observed=True):
            if len(sub) < min_size:
                continue

            label = _label_group_name(name)
            y = sub[y_col].to_numpy(dtype=float)
            expected = np.asarray(benchmark_predict(benchmark_fit_resolved, sub, x_col=x_col), dtype=float)
            residual = y - expected
            residual_pct = np.where(expected != 0, residual / expected * 100, np.nan)

            median_residual = float(np.median(residual))
            mad_val = float(1.4826 * np.median(np.abs(residual - median_residual)))
            safe_mad = mad_val if mad_val > 0 else np.inf
            z = (residual - median_residual) / safe_mad
            threshold = k * mad_val

            if direction == "negative":
                flagged = z < -k
            elif direction == "positive":
                flagged = z > k
            else:
                flagged = np.abs(z) > k

            if id_cols:
                block = sub[id_cols].reset_index(drop=True)
            else:
                block = pd.DataFrame({"row_id": sub.index})

            block["actual"] = y
            block["expected"] = expected
            block["residual"] = residual
            block["residual_pct"] = residual_pct
            block["segment"] = label
            block["segment_level"] = level
            block["segment_mad"] = mad_val
            block["threshold"] = threshold
            block["flagged"] = flagged

            blocks.append(block)

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def _is_reportable_scalar(value):
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return False
    return isinstance(value, (int, float, str, bool, np.integer, np.floating, np.bool_))


def segment_stability_drilldown_report(df, x_col, y_col, segment_cols, min_size=20, min_points=5, degree=2, **stability_kwargs):
    """
    Report model_stability_pct results at EVERY level of the hierarchy
    simultaneously, without exclusive assignment -- the same
    individual can be part of multiple rows here (e.g. once within a
    JobFamily x Level x OT group, and again within the broader
    Level x OT group), whenever both groupings independently meet
    min_size.

    Complementary to segment_stability_report (which assigns each
    individual to exactly one, most-specific segment): this answers
    "how does model stability change as segment granularity changes?"
    rather than "what is the single most relevant segment for this
    individual?" -- the third drilldown function, alongside
    segment_benchmark_drilldown_report and mad_outlier_drilldown_report.

    Mirrors apply_by_segment's row structure (segment, n, skipped,
    reason/error, then the analysis function's scalar results) at
    every hierarchy level, with an added segment_level column.
    """
    hierarchy = _build_hierarchy(list(segment_cols))
    rows = []

    for level, cols in enumerate(hierarchy):
        for name, sub in df.groupby(cols, observed=True):
            if len(sub) < min_size:
                continue

            label = _label_group_name(name)
            row = {"segment": label, "segment_level": level, "n": len(sub)}

            if len(sub) < min_points:
                row["skipped"] = True
                row["reason"] = f"fewer than {min_points} points"
                rows.append(row)
                continue

            row["skipped"] = False

            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)

            try:
                result = model_stability_pct(x, y, degree=degree, **stability_kwargs)
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                rows.append(row)
                continue

            for key, value in result.items():
                if _is_reportable_scalar(value):
                    row[key] = value

            rows.append(row)

    return pd.DataFrame(rows).sort_values(["segment_level", "segment"]).reset_index(drop=True)


def outlier_drilldown_summary(drilldown_report, id_col="row_id"):
    """
    Given the output of mad_outlier_drilldown_report, count how many
    hierarchy LEVELS each individual was flagged on -- a rough
    "Cook's impact for segmentation" measure. An individual flagged
    across many levels (e.g. at the finest JobFamily x Level x OT
    grouping AND at every broader fallback) has a conclusion that
    survives changing the reference population -- a stronger signal
    than one that only appears as an outlier under one specific,
    narrow definition of "expected".

    id_col: the identifier column present in drilldown_report (e.g.
        an employee ID column passed as id_cols to
        mad_outlier_drilldown_report, or "row_id" if none was given).

    Returns a DataFrame with one row per flagged individual:
    id_col, outlier_levels (count of hierarchy levels flagged on),
    and levels (the sorted list of segment_level values it was flagged
    at) -- individuals never flagged at any level are not included.
    """
    flagged = drilldown_report[drilldown_report["flagged"]]
    if len(flagged) == 0:
        return pd.DataFrame(columns=[id_col, "outlier_levels", "levels"])

    grouped = flagged.groupby(id_col)["segment_level"]
    summary = grouped.agg(outlier_levels="count", levels=lambda s: sorted(s.tolist()))
    return summary.reset_index().sort_values("outlier_levels", ascending=False).reset_index(drop=True)


def segment_quality_report(df, x_col, y_col, segment_cols, min_size=20, min_points=5, degree=2):
    """
    Combine several independent robustness/quality measures into a
    single row per segment: model stability across fitting methods,
    Cook's impact of influential points, MdAPE (median absolute
    percentage error of the Huber fit), and the IQR of residuals --
    so a single glance at one table shows how trustworthy each
    segment's conclusion is, rather than needing to run several
    separate functions and cross-reference them by hand.

    Uses the same EXCLUSIVE hierarchical segmentation as
    segment_stability_report / segment_benchmark_report (each
    individual assigned to exactly one, most-specific segment meeting
    min_size).

    Each measure is computed independently and wrapped in its own
    error handling: a Tukey-based stability failure (e.g. statsmodels
    not installed) does not prevent Cook's impact, MdAPE, or IQR from
    still being reported for that segment.

    Returns one row per segment with: segment, segment_level, n,
    skipped, median_pct_diff / max_pct_diff (from model_stability_pct,
    or stability_error if that failed), n_flagged / cook_impact_pct
    (from Cook's diagnostics, or cook_error if that failed), and
    mdape / iqr_resid (from a fresh Huber fit, or fit_error if that
    failed).
    """
    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    rows = []
    for segment_value, group in segmented.groupby("segment_id", observed=True):
        row = {"segment": segment_value, "n": len(group)}

        if len(group) < min_points:
            row["skipped"] = True
            rows.append(row)
            continue
        row["skipped"] = False

        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)

        try:
            stability = model_stability_pct(x, y, degree=degree)
            row["median_pct_diff"] = stability.get("median_pct_diff")
            row["max_pct_diff"] = stability.get("max_pct_diff")
        except Exception as exc:  # noqa: BLE001
            row["stability_error"] = str(exc)

        try:
            diag = cooks_diagnostic(x, y, degree=degree)
            row["n_flagged"] = len(diag["flagged_indices"])
            if len(diag["flagged_indices"]) > 0:
                impact = cook_impact(x, y, diag["flagged_indices"], degree=degree)
                row["cook_impact_pct"] = impact["median_pct_change"]
            else:
                row["cook_impact_pct"] = 0.0
        except Exception as exc:  # noqa: BLE001
            row["cook_error"] = str(exc)

        try:
            fit = fit_huber_trend(x, y, degree=degree)
            pred = predict_trend(fit, x)
            residuals = y - pred
            ape = np.abs(residuals / np.where(y != 0, y, np.nan)) * 100
            row["mdape"] = float(np.nanmedian(ape))
            row["iqr_resid"] = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        except Exception as exc:  # noqa: BLE001
            row["fit_error"] = str(exc)

        rows.append(row)

    result = pd.DataFrame(rows)
    level_map = segmented.groupby("segment_id")["segment_level"].first()
    result["segment_level"] = result["segment"].map(level_map)
    return result



def dual_reference_outlier_report(df, y_col, x_col, segment_cols, benchmark_fit=None, degree=2,
                                   min_size=20, k=3.0, direction="negative", id_cols=None):
    """
    Classify each individual by whether their residual is a MAD
    outlier relative to TWO different reference populations
    simultaneously:

      - LOCAL: their own segment's Huber trend, fit fresh on just that
        segment's (x_col, y_col), with the MAD threshold computed
        WITHIN that same segment. Answers "how does this person
        deviate from their immediate colleagues?"
      - GLOBAL: the benchmark model (default single-column Huber
        trend, or any custom multi-column model via benchmark_fit),
        with the MAD threshold computed across the ENTIRE population
        -- not re-centered per segment. Answers "how does this person
        deviate from the organization's expected structure, including
        whether their whole segment is collectively off?"

    This population-wide (not per-segment) MAD for the global
    reference is deliberate and important: if it were re-centered per
    segment (as mad_outlier_report's flagging is, by design, for its
    own single-reference use case), a segment that is UNIFORMLY
    shifted relative to the benchmark would have that shift silently
    absorbed by the per-segment median subtraction -- nobody in it
    would ever be flagged globally, no matter how far the whole
    segment sits from the benchmark, since everyone would share
    roughly the same "typical" residual for their segment. Computing
    the global MAD across the whole population instead means a
    segment that is collectively low CAN surface as a global
    deviation, distinct from an individual deviation within an
    otherwise-typical segment.

    This is "Reference Population Sensitivity" -- a fourth robustness
    axis, alongside model choice (Huber/Tukey/OLS in core), individual
    observations (Cook's impact), and segment granularity (the
    drilldown functions): does a conclusion about a specific
    individual survive changing WHICH reference population they are
    compared against?

    The `reference_type` column classifies each individual:
      "A" -- outlier under BOTH references. The strongest candidates:
             low relative to colleagues AND low relative to the
             organization's expected structure, regardless of how
             "expected" is defined.
      "B" -- outlier LOCALLY only. Often means their segment has tight
             internal spread (so a modest dip stands out among close
             peers), even though the same dip isn't unusual against
             the wider, noisier population.
      "C" -- outlier GLOBALLY only. Often means the whole segment
             trends low relative to the benchmark, but this individual
             is typical within their own segment.
      "D" -- outlier under neither reference (the normal case).

    Returns one row per individual: id_cols (or row_id), actual,
    local_expected, local_residual, local_flagged, global_expected,
    global_residual, global_flagged, reference_type, segment,
    segment_level.
    """
    if direction not in ("negative", "positive", "two_sided"):
        raise ValueError(f"direction must be 'negative', 'positive', or 'two_sided', got {direction!r}")

    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)
    y_values = segmented[y_col].to_numpy(dtype=float)

    def _flag(residual, z):
        if direction == "negative":
            return z < -k
        if direction == "positive":
            return z > k
        return np.abs(z) > k

    # ---- LOCAL: fresh per-segment Huber fit, MAD centered WITHIN each segment ----
    local_expected = np.full(len(segmented), np.nan, dtype=float)
    for segment_value, group in segmented.groupby("segment_id", observed=True):
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        try:
            local_fit = fit_huber_trend(x, y, degree=degree)
            positions = segmented.index.get_indexer(group.index)
            local_expected[positions] = predict_trend(local_fit, x)
        except Exception:  # noqa: BLE001 -- leave as NaN; flagged becomes False for these rows
            pass

    local_residual = y_values - local_expected

    segmented = segmented.copy()
    segmented["_local_residual"] = local_residual
    seg_median_local = segmented.groupby("segment_id")["_local_residual"].median()
    seg_mad_local = segmented.groupby("segment_id")["_local_residual"].apply(
        lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))
    )
    median_local_per_row = segmented["segment_id"].map(seg_median_local).to_numpy(dtype=float)
    mad_local_per_row = segmented["segment_id"].map(seg_mad_local).to_numpy(dtype=float)
    safe_mad_local = np.where(mad_local_per_row > 0, mad_local_per_row, np.inf)
    z_local = (local_residual - median_local_per_row) / safe_mad_local
    local_flagged = np.nan_to_num(_flag(local_residual, z_local), nan=0.0).astype(bool)

    # ---- GLOBAL: benchmark expected values, MAD centered across the WHOLE population ----
    if benchmark_fit is None:
        if x_col is None:
            raise ValueError("x_col must be supplied when no custom benchmark model is provided.")
        from ..benchmark.global_model import fit_huber_benchmark
        benchmark_fit_resolved = fit_huber_benchmark(
            df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree,
        )
    else:
        benchmark_fit_resolved = benchmark_fit

    from ..benchmark.global_model import benchmark_predict
    global_expected = np.asarray(benchmark_predict(benchmark_fit_resolved, segmented, x_col=x_col), dtype=float)
    global_residual = y_values - global_expected

    global_median = float(np.median(global_residual))
    global_mad = float(1.4826 * np.median(np.abs(global_residual - global_median)))
    safe_global_mad = global_mad if global_mad > 0 else np.inf
    z_global = (global_residual - global_median) / safe_global_mad
    global_flagged = _flag(global_residual, z_global)

    if id_cols:
        result = segmented[id_cols].reset_index(drop=True)
    else:
        result = pd.DataFrame({"row_id": segmented.index})

    result["actual"] = y_values
    result["local_expected"] = local_expected
    result["local_residual"] = local_residual
    result["local_flagged"] = local_flagged
    result["global_expected"] = global_expected
    result["global_residual"] = global_residual
    result["global_flagged"] = global_flagged
    result["segment"] = segmented["segment_id"].to_numpy()
    result["segment_level"] = segmented["segment_level"].to_numpy()

    conditions = [
        result["local_flagged"] & result["global_flagged"],
        result["local_flagged"] & ~result["global_flagged"],
        ~result["local_flagged"] & result["global_flagged"],
    ]
    choices = ["A", "B", "C"]
    result["reference_type"] = np.select(conditions, choices, default="D")

    return result


def export_huber_iqr_pdf(df, x_col, y_col, segment_cols, path, min_size=20, min_points_to_plot=5,
                          title_fn=None, degree=2, bins=15, grouping="bin", min_n_for_iqr=5,
                          methods=("huber",), show_bootstrap_band=False, bootstrap_levels=(95,),
                          n_boot="auto", cap_style="matplotlib", residual_box_metric="r2",
                          ylim="auto", show_undersized_points=True, style=None, figsize=(10, 6)):
    """
    Render one plot_huber_iqr chart per segment -- built from the same
    automatic hierarchical segmentation as the rest of
    robustkit.segment_awareness -- as a one-page-per-segment PDF.

    This is the "member-facing" counterpart to export_outlier_pdf
    (which is typically for internal review): a full trend-plus-spread
    chart per segment, suitable for sharing with the people the chart
    actually describes.

    Segments with fewer than min_points_to_plot observations are
    skipped (matching export_outlier_pdf's convention -- a chart with
    a handful of points isn't meaningfully readable).

    title_fn: optional callable(segment_name, info) -> str, where info
        is a dict with {"n", "segment_level"}. Lets callers apply
        their own domain-specific segment naming (e.g. translating an
        internal code like "ENG_P3_Yes" into a readable label such as
        "Engineers, level 3, overtime-eligible") without robustkit
        needing to know anything about that domain. Defaults to a
        generic "{segment_name} (n=...)" title if not given.

    All other parameters (degree, bins, grouping, min_n_for_iqr,
    methods, show_bootstrap_band, bootstrap_levels, n_boot, cap_style,
    residual_box_metric, ylim, show_undersized_points, style, figsize)
    are passed straight through to plot_huber_iqr for each segment's
    page -- see that function's docstring for what they do and how to
    configure them (e.g. to match an existing chart style exactly).

    Returns `path`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    hierarchy = _build_hierarchy(list(segment_cols))
    segmented = hierarchical_segment(df, hierarchy, min_size=min_size)

    with PdfPages(path) as pdf:
        for segment_value, group in segmented.groupby("segment_id", observed=True):
            if len(group) < min_points_to_plot:
                continue

            x = group[x_col].to_numpy(dtype=float)
            y = group[y_col].to_numpy(dtype=float)

            if title_fn is not None:
                info = {"n": len(group), "segment_level": int(group["segment_level"].iloc[0])}
                title = title_fn(segment_value, info)
            else:
                title = f"{segment_value} (n={len(group)})"

            fig, ax = plt.subplots(figsize=figsize)

            plot_huber_iqr(
                x, y, degree=degree, bins=bins, grouping=grouping, min_n_for_iqr=min_n_for_iqr,
                show_points=False, show_residual_box=True, residual_box_metric=residual_box_metric,
                methods=methods, show_bootstrap_band=show_bootstrap_band, bootstrap_levels=bootstrap_levels,
                n_boot=n_boot, cap_style=cap_style, ylim=ylim, show_undersized_points=show_undersized_points,
                style=style, title=title, ax=ax, figsize=figsize,
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return path

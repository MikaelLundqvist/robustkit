"""
Validation runner for robustkit.segment_awareness: segment_stability_report,
segment_benchmark_report, mad_outlier_report, and export_outlier_pdf --
all built on automatic hierarchical segmentation.

Uses each dataset's `hierarchy_cols` (curated categorical columns,
ordered most-specific-first) rather than `feature_cols`, since a
meaningful hierarchy needs multiple categorical grouping dimensions,
not the mostly-continuous columns used elsewhere.
"""

import tempfile
from pathlib import Path

import pandas as pd

import robustkit as rk

from .common import section, safe_run


def run_segment_awareness_module(df, x_col, y_col, hierarchy_cols):
    section(f"robustkit.segment_awareness -- hierarchy_cols={hierarchy_cols}")

    if not hierarchy_cols:
        print("  (skipped -- no hierarchy_cols defined for this dataset)")
        return

    cols = list(dict.fromkeys([x_col, y_col] + hierarchy_cols))
    sub = df[cols].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    for c in hierarchy_cols:
        sub[c] = sub[c].astype(str)
    sub = sub.dropna()

    # ---- segment_stability_report ----
    stability = safe_run(
        f"segment_stability_report (segment_cols={hierarchy_cols})",
        lambda: rk.segment_stability_report(sub, x_col=x_col, y_col=y_col, segment_cols=hierarchy_cols, min_size=20),
    )
    if stability is not None:
        cols_to_show = [c for c in ["segment", "n", "skipped", "median_pct_diff", "segment_level"] if c in stability.columns]
        print(stability[cols_to_show].to_string(index=False))
        levels_used = sorted(stability["segment_level"].dropna().unique().tolist())
        print(f"\n  Hierarchy levels actually used: {levels_used} "
              f"(0 = finest = {hierarchy_cols}; higher = fallback to a coarser grouping)")

    # ---- segment_benchmark_report ----
    benchmark = safe_run(
        f"segment_benchmark_report (segment_cols={hierarchy_cols})",
        lambda: rk.segment_benchmark_report(sub, y_col=y_col, segment_cols=hierarchy_cols, x_col=x_col, min_size=20, n_boot=100),
    )
    if benchmark is not None:
        print(benchmark.to_string(index=False))

    # ---- mad_outlier_report ----
    outliers = safe_run(
        f"mad_outlier_report (segment_cols={hierarchy_cols})",
        lambda: rk.mad_outlier_report(sub, y_col=y_col, segment_cols=hierarchy_cols, x_col=x_col, min_size=20, k=3.0),
    )
    if outliers is not None:
        n_flagged = int(outliers["flagged"].sum())
        print(f"  Total individuals: {len(outliers)}, flagged as outliers: {n_flagged}")
        if n_flagged > 0:
            print(outliers[outliers["flagged"]].sort_values("residual").head(10).to_string(index=False))

    # ---- export_outlier_pdf ----
    pdf_path = str(Path(tempfile.gettempdir()) / "validation_outliers.pdf")
    pdf_result = safe_run(
        "export_outlier_pdf",
        lambda: rk.export_outlier_pdf(
            sub, y_col=y_col, segment_cols=hierarchy_cols, x_col=x_col, path=pdf_path,
            min_size=20, k=3.0, min_points_to_plot=5,
        ),
    )
    if pdf_result:
        print(f"  PDF written to: {pdf_result}")

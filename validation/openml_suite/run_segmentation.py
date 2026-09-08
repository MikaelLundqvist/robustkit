"""
Validation runner for robustkit.segmentation.
"""

import pandas as pd

import robustkit as rk

from .common import section, safe_run


def run_segmentation_module(df, x_col, y_col, segment_col):
    section(f"robustkit.segmentation -- segment={segment_col}")
    if segment_col is None:
        print("  (skipped -- no segment column defined for this dataset)")
        return

    sub = df[[x_col, y_col, segment_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    sub[segment_col] = sub[segment_col].astype(str)

    segmented = safe_run("hierarchical_segment", lambda: rk.hierarchical_segment(sub, [[segment_col]], min_size=20))
    if segmented is not None:
        print("  segment sizes:")
        print("   ", rk.segment_sizes(segmented).to_string().replace("\n", "\n    "))

        report = safe_run(
            "apply_by_segment(model_stability_pct)",
            lambda: rk.apply_by_segment(segmented, "segment_id", x_col, y_col, rk.model_stability_pct),
        )
        if report is not None:
            cols = [c for c in ["segment", "n", "skipped", "median_pct_diff"] if c in report.columns]
            print(report[cols].to_string(index=False))

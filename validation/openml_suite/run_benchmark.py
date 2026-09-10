"""
Validation runner for robustkit.benchmark: segment_position_report,
feature_robustness_report, and the Fase B reporting layer (residual_summary,
deviation_report, benchmark_report_suite, export_benchmark_excel).

Deliberately does NOT pre-filter small segments before calling
segment_position_report -- that used to mask the MIN_POINTS_FOR_CI fix
(GAP #4) by removing exactly the segments it exists to handle. Small
segments are left in; the report itself should show ci_available=False
for them without crashing.
"""

import tempfile
from pathlib import Path

import pandas as pd

import robustkit as rk

from .common import section, safe_run


def run_benchmark_module(df, x_col, y_col, segment_col, feature_cols, hierarchy_cols=None):
    section(f"robustkit.benchmark -- segment={segment_col}")

    hierarchy_cols = hierarchy_cols or ([segment_col] if segment_col else [])
    cols = list(dict.fromkeys([x_col, y_col] + feature_cols + hierarchy_cols + ([segment_col] if segment_col else [])))
    sub = df[cols].copy()
    for c in [x_col, y_col] + feature_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    for c in hierarchy_cols:
        if c in sub.columns:
            sub[c] = sub[c].astype(str)
    sub = sub.dropna()

    if segment_col is not None:
        sub[segment_col] = sub[segment_col].astype(str)

        # No pre-filtering of small segments here -- segment_position_report
        # must handle whatever segment sizes actually occur (GAP #4).
        report = safe_run(
            "segment_position_report (small segments included, not pre-filtered)",
            lambda: rk.segment_position_report(sub, segment_col=segment_col, x_col=x_col, y_col=y_col, n_boot=200),
        )
        if report is not None:
            print(report.to_string(index=False))
            n_small = int((~report["ci_available"]).sum())
            n_large = int(report["ci_available"].sum())
            print(f"\n  {n_large} segment(s) with ci_available=True, "
                  f"{n_small} segment(s) with ci_available=False (n < MIN_POINTS_FOR_CI={rk.MIN_POINTS_FOR_CI})")

            # ---- Fase B: residual_summary ----
            rs = safe_run(
                "residual_summary",
                lambda: rk.residual_summary(sub, segment_col=segment_col, y_col=y_col, x_col=x_col),
            )
            if rs is not None:
                print(rs.to_string(index=False))

            # ---- Fase B: deviation_report ----
            ndr = safe_run(
                "deviation_report",
                lambda: rk.deviation_report(sub, y_col=y_col, x_col=x_col, top_n=10),
            )
            if ndr is not None:
                print(ndr.to_string(index=False))
    else:
        print("  (skipped segment_position_report / residual_summary / deviation_report "
              "-- no segment column defined for this dataset)")

    # ---- Fase B: benchmark_report_suite + export_benchmark_excel ----
    # Uses hierarchy_cols (curated categorical columns) as multiple
    # grouping columns -- feature_cols are mostly continuous and
    # wouldn't exercise the "multiple grouping columns" case at all.
    group_columns = [c for c in hierarchy_cols if c in sub.columns]

    if len(group_columns) >= 1:
        suite = safe_run(
            f"benchmark_report_suite (group_columns={group_columns})",
            lambda: rk.benchmark_report_suite(sub, group_columns=group_columns, y_col=y_col, x_col=x_col, n_boot=100),
        )
        if suite is not None:
            for name, rep in suite.items():
                print(f"--- benchmark_report_suite[{name}] ---")
                print(rep.to_string(index=False))

            excel_result = safe_run(
                "export_benchmark_excel",
                lambda: rk.export_benchmark_excel(suite, str(Path(tempfile.gettempdir()) / "validation_benchmark_report.xlsx")),
            )
            if excel_result:
                print(f"  Excel report written to: {excel_result}")
    else:
        print("  (skipped benchmark_report_suite / export_benchmark_excel -- no grouping columns available)")

    # ---- feature_robustness_report (unchanged from before, still capped
    # to the first 4 features for runtime reasons -- see run_information.py) ----
    rob_features = list(dict.fromkeys([x_col] + [c for c in feature_cols if c != x_col]))
    rob_features = [c for c in rob_features if c in sub.columns][:4]

    if len(rob_features) < 2:
        print(f"  (skipped feature_robustness_report -- fewer than 2 usable features: {rob_features})")
        return

    rob_report = safe_run(
        f"feature_robustness_report (features={rob_features})",
        lambda: rk.feature_robustness_report(sub, target=y_col, features=rob_features),
    )
    if rob_report is not None:
        print(rob_report.to_string(index=False))

"""
Validation runner for robustkit.benchmark.
"""

import pandas as pd

import robustkit as rk

from .common import section, safe_run


def run_benchmark_module(df, x_col, y_col, segment_col, feature_cols):
    section(f"robustkit.benchmark -- segment={segment_col}")

    cols = list(dict.fromkeys([x_col, y_col] + feature_cols + ([segment_col] if segment_col else [])))
    sub = df[cols].copy()
    for c in [x_col, y_col] + feature_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    if segment_col is not None:
        sub[segment_col] = sub[segment_col].astype(str)
        sizes = sub[segment_col].value_counts()
        keep = sizes[sizes >= 20].index
        seg_sub = sub[sub[segment_col].isin(keep)]

        if seg_sub[segment_col].nunique() < 2:
            print("  (skipped segment_position_report -- fewer than 2 segments have >= 20 rows)")
        else:
            report = safe_run(
                "segment_position_report",
                lambda: rk.segment_position_report(seg_sub, segment_col=segment_col, x_col=x_col, y_col=y_col, n_boot=200),
            )
            if report is not None:
                print(report.to_string(index=False))
    else:
        print("  (skipped segment_position_report -- no segment column defined for this dataset)")

    # feature_robustness_report needs >= 2 features to classify meaningfully
    # (see its docstring) and does NOT need a segment column at all.
    # NOTE: capped to 4 features for runtime reasons -- for datasets
    # with more candidate features (e.g. Boston's 13), this means not
    # every feature gets classified. Known limitation, revisit if a
    # full sweep is needed.
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

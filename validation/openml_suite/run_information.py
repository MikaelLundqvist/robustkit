"""
Validation runner for robustkit.information.
"""

import robustkit as rk

from .common import section, safe_run


def run_information_module(df, y_col, feature_cols):
    section(f"robustkit.information -- target={y_col}")
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print(f"  (skipped -- none of the expected feature columns {feature_cols} found)")
        return

    sub = df[cols + [y_col]].copy().dropna()

    ranking = safe_run("rank_features", lambda: rk.rank_features(sub, target=y_col))
    if ranking is not None:
        print(ranking.to_string(index=False))

    report = safe_run("quadrant_report", lambda: rk.quadrant_report(sub, target=y_col))
    if report is not None:
        print(report[["feature", "quadrant"]].to_string(index=False))

    # NOTE: rank_communicative_pairs is capped to the first 4 feature
    # columns for runtime reasons (it's O(k^2) in the number of
    # features). For datasets with more candidate features (e.g.
    # Boston's 13), this means not every pair gets evaluated -- known
    # limitation, revisit if a full pairwise sweep is needed.
    if len(cols) >= 2:
        pairs = safe_run(
            "rank_communicative_pairs",
            lambda: rk.rank_communicative_pairs(sub, target=y_col, features=cols[:4]),
        )
        if pairs is not None:
            print(pairs[["feature_1", "feature_2", "redundancy", "synergy", "pair_score"]].to_string(index=False))

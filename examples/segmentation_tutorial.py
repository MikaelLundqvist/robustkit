"""
robustkit.segmentation quickstart tutorial
=============================================

A self-contained walkthrough of hierarchical segmentation and
per-segment analysis, run against a small synthetic dataset with a
deliberately small subgroup (to show the min-size fallback in action).
Run with:

    python examples/segmentation_tutorial.py
"""

import numpy as np
import pandas as pd

from robustkit import (
    hierarchical_segment, segment_sizes, apply_by_segment,
    model_stability_pct, cooks_diagnostic, cook_impact,
)


def make_example_data(n=400, seed=0):
    """
    A dataset with three grouping columns (department, level, status)
    and one deliberately tiny combination -- "Ops" x "L3" -- too small
    to analyze on its own, so hierarchical_segment has to fall back to
    a coarser grouping for those rows.
    """
    rng = np.random.default_rng(seed)

    department = rng.choice(["Engineering", "Sales", "Ops"], size=n, p=[0.55, 0.35, 0.10])
    level = rng.choice(["L1", "L2", "L3"], size=n, p=[0.5, 0.35, 0.15])
    status = rng.choice(["Active", "OnLeave"], size=n, p=[0.85, 0.15])

    x = rng.uniform(20, 60, n)
    y = 1000 + 45 * x - 0.3 * x**2 + rng.normal(0, 250, n)

    df = pd.DataFrame({
        "department": department, "level": level, "status": status,
        "x": x, "y": y,
    })

    # Shrink Ops/L3 down to just a few rows, to guarantee a fallback case
    ops_l3 = df[(df["department"] == "Ops") & (df["level"] == "L3")]
    if len(ops_l3) > 5:
        drop_idx = ops_l3.index[5:]
        df = df.drop(index=drop_idx).reset_index(drop=True)

    return df


def main():
    df = make_example_data()
    print(f"Example dataset: {len(df)} rows\n")

    # ------------------------------------------------------------------
    # 1. Hierarchical segmentation with fallback
    # ------------------------------------------------------------------
    print("=== 1. Hierarchical segmentation ===")
    hierarchy = [
        ["department", "level", "status"],  # finest
        ["department", "level"],
        ["department"],                     # coarsest
    ]
    segmented = hierarchical_segment(df, hierarchy, min_size=20)

    print("Segment sizes (min_size=20):")
    print(segment_sizes(segmented))
    print()

    fallback_rows = segmented[segmented["segment_level"] > 0]
    print(f"{len(fallback_rows)} rows fell back to a coarser grouping "
          f"because their finest-grained segment was too small.\n")

    # ------------------------------------------------------------------
    # 2. Run model_stability_pct independently per segment
    # ------------------------------------------------------------------
    print("=== 2. Model stability per segment ===")
    stability_report = apply_by_segment(
        segmented, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=model_stability_pct,
    )
    print(stability_report[["segment", "n", "skipped", "median_pct_diff", "max_pct_diff"]]
          .to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 3. A custom analysis function: Cook-impact per segment
    # ------------------------------------------------------------------
    print("=== 3. Custom analysis: Cook-impact per segment ===")

    def cook_impact_summary(x, y):
        diag = cooks_diagnostic(x, y)
        if len(diag["flagged_indices"]) == 0:
            return {"n_flagged": 0, "median_pct_change": 0.0}
        impact = cook_impact(x, y, diag["flagged_indices"])
        return {
            "n_flagged": len(diag["flagged_indices"]),
            "median_pct_change": impact["median_pct_change"],
        }

    impact_report = apply_by_segment(
        segmented, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=cook_impact_summary, min_points=10,
    )
    print(impact_report[["segment", "n", "skipped", "n_flagged", "median_pct_change"]]
          .to_string(index=False))
    print()
    print("apply_by_segment works with ANY analysis_fn(x, y) -> dict --")
    print("built-in robustkit functions or your own, as shown above.")


if __name__ == "__main__":
    main()

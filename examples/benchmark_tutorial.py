"""
robustkit.benchmark quickstart tutorial
==========================================

A self-contained walkthrough of benchmarking segments against a global
robust trend, and mapping features by how trustworthy their
relationship with the target is. Run with:

    python examples/benchmark_tutorial.py
"""

import numpy as np
import pandas as pd

from robustkit import (
    fit_huber_benchmark, segment_position_report,
    feature_robustness_report, plot_feature_robustness,
)


def make_segment_example_data(n=600, seed=1):
    """
    Three departments sharing the same age->salary trend shape, but
    HR is shifted below it and Finance above it -- IT sits right on
    the global trend.
    """
    rng = np.random.default_rng(seed)
    department = rng.choice(["Engineering", "HR", "Finance"], size=n, p=[0.4, 0.3, 0.3])
    age = rng.uniform(25, 60, n)

    shift = np.select(
        [department == "HR", department == "Finance"],
        [-1200, 400],
        default=0,
    )
    salary = 30000 + 400 * age + shift + rng.normal(0, 800, n)

    return pd.DataFrame({"department": department, "age": age, "salary": salary})


def make_robustness_example_data(n=300, seed=2):
    """
    Two features sharing one target: one well-behaved, one with a
    handful of genuinely influential points injected (extreme x AND a
    target value inconsistent with that feature's own trend).

    Both features share the same target column (feature_robustness_report
    compares multiple features against one target), so the trick is
    giving each feature *different* x values at the rows where the
    shared target looks "unusual" -- x_stable's values there are
    chosen to still be consistent with its own trend, x_fragile's are
    not.
    """
    rng = np.random.default_rng(seed)
    x_base = rng.uniform(0, 10, n)
    target = 5 + 2 * x_base + rng.normal(0, 1, n)

    # A few rows where the shared target takes values that look
    # "unusual" relative to a naive continuation of the trend.
    extra_target = np.array([5.0, 3.0, 8.0])

    # x_fragile: genuinely extreme x at those rows, AND the target
    # value there breaks x_fragile's own trend -- a true influential
    # point (high leverage + high residual).
    x_fragile = np.concatenate([x_base, [25.0, 27.0, 30.0]])

    # x_stable: at those SAME rows, take whatever x value would
    # actually be consistent with x_stable's trend given the target
    # value there -- so this feature stays perfectly on-trend
    # throughout, despite the target being "unusual" at those rows.
    x_stable_extra = (extra_target - 5) / 2
    x_stable = np.concatenate([x_base, x_stable_extra])

    target_full = np.concatenate([target, extra_target])

    return pd.DataFrame({"x_stable": x_stable, "x_fragile": x_fragile, "target": target_full})


def main():
    # ------------------------------------------------------------------
    # 1. Benchmark segments against a single global trend
    # ------------------------------------------------------------------
    print("=== 1. Segment position vs. global benchmark ===")
    seg_df = make_segment_example_data()

    benchmark_fit = fit_huber_benchmark(
        seg_df["age"].to_numpy(dtype=float), seg_df["salary"].to_numpy(dtype=float),
    )

    report = segment_position_report(
        seg_df, segment_col="department", x_col="age", y_col="salary",
        benchmark_fit=benchmark_fit, n_boot=300,
    )
    print(report.to_string(index=False))
    print()
    print("Interpretation: a segment's CI crossing zero means no clear")
    print("deviation from the overall trend. HR and Finance don't cross")
    print("zero (clearly below / above); Engineering does (on-trend).\n")

    # ------------------------------------------------------------------
    # 2. Map features by robustness (stability x Cook-impact)
    # ------------------------------------------------------------------
    print("=== 2. Feature robustness map ===")
    rob_df = make_robustness_example_data()

    rob_report = feature_robustness_report(rob_df, target="target")
    print(rob_report.to_string(index=False))
    print()
    print("x_fragile has injected influential points (extreme x, trend-")
    print("inconsistent y) and should show high cook_impact_pct; x_stable")
    print("shares the same extreme x values but with trend-consistent y,")
    print("so it should NOT be flagged as fragile.\n")

    print("Calling plot_feature_robustness(report=rob_report) opens a")
    print("matplotlib figure with features placed in (stability, Cook-")
    print("impact) space, colored by quadrant -- always consistent with")
    print("the table above.")
    # Uncomment to actually display the plot:
    # plot_feature_robustness(report=rob_report)


if __name__ == "__main__":
    main()

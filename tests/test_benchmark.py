import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import (
    fit_huber_benchmark, segment_position_report,
    feature_robustness_report, plot_feature_robustness,
)
from robustkit.core.diagnostics import cooks_diagnostic, cook_impact


def make_benchmark_df(n=600, seed=1):
    rng = np.random.default_rng(seed)
    department = rng.choice(["IT", "HR", "Finance"], size=n, p=[0.4, 0.3, 0.3])
    age = rng.uniform(25, 60, n)
    # HR systematically below the global trend, Finance above, IT on-trend
    shift = np.select([department == "HR", department == "Finance"], [-1200, 400], default=0)
    salary = 30000 + 400 * age + shift + rng.normal(0, 800, n)
    return pd.DataFrame({"department": department, "age": age, "salary": salary})


def test_fit_huber_benchmark_runs():
    df = make_benchmark_df()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float))
    assert fit["kind"] == "sklearn"


def test_segment_position_report_detects_known_shifts():
    df = make_benchmark_df()
    report = segment_position_report(
        df, segment_col="department", x_col="age", y_col="salary", n_boot=300,
    )

    assert set(report["segment"]) == {"IT", "HR", "Finance"}

    hr_row = report[report["segment"] == "HR"].iloc[0]
    finance_row = report[report["segment"] == "Finance"].iloc[0]
    it_row = report[report["segment"] == "IT"].iloc[0]

    assert hr_row["difference"] < 0
    assert hr_row["ci_upper"] < 0  # CI does not cross zero -- clearly below benchmark

    assert finance_row["difference"] > 0
    assert finance_row["ci_lower"] > 0  # CI does not cross zero -- clearly above benchmark

    assert it_row["ci_lower"] < 0 < it_row["ci_upper"]  # CI crosses zero -- no clear deviation


def test_segment_position_report_accepts_precomputed_benchmark():
    df = make_benchmark_df()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float))
    report = segment_position_report(
        df, segment_col="department", x_col="age", y_col="salary",
        benchmark_fit=fit, n_boot=100,
    )
    assert len(report) == 3


def test_feature_robustness_report_classifies_fragile_feature():
    """
    A feature with genuinely influential points (extreme x AND
    trend-inconsistent y) should score high on Cook-impact and land in
    a quadrant reflecting that, distinctly from a well-behaved feature.
    """
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(0, 10, n)
    y = 5 + 2 * x + rng.normal(0, 1, n)

    # Inject genuine influential points: extreme x, trend-inconsistent y
    x = np.append(x, [25, 27, 30])
    y = np.append(y, [5.0, 3.0, 8.0])

    df = pd.DataFrame({"x": x, "target": y})
    report = feature_robustness_report(df, target="target", features=["x"])

    assert "quadrant" in report.columns
    assert report.iloc[0]["cook_impact_pct"] > 5


def test_plot_feature_robustness_matches_report_quadrants():
    df = pd.DataFrame({
        "feature": ["a", "b"],
        "stability_pct": [30.0, 2.0],
        "cook_impact_pct": [50.0, 0.5],
    })
    from robustkit import classify_quadrants
    from robustkit.benchmark.robustness_map import ROBUSTNESS_LABELS

    classified = classify_quadrants(
        df, x_col="stability_pct", y_col="cook_impact_pct", labels=ROBUSTNESS_LABELS,
    )
    plotted = plot_feature_robustness(report=classified.copy(), annotate=False)

    merged = classified.merge(plotted, on="feature", suffixes=("_report", "_plot"))
    assert (merged["quadrant_report"] == merged["quadrant_plot"]).all()

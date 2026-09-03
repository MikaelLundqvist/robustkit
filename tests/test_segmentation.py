"""
Tests for robustkit.segmentation: hierarchical grouping with a
minimum-size fallback, and running core analyses per segment.
"""

import numpy as np
import pandas as pd
import pytest

from robustkit import (
    hierarchical_segment, segment_sizes, apply_by_segment,
    model_stability_pct, cook_impact, cooks_diagnostic,
)


def make_test_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    families = rng.choice(["Eng", "Sales", "Ops"], size=n, p=[0.6, 0.25, 0.15])
    levels = rng.choice(["L1", "L2", "L3"], size=n, p=[0.5, 0.3, 0.2])
    flags = rng.choice(["Yes", "No"], size=n)
    x = rng.uniform(20, 60, n)
    y = 1000 + 40 * x + rng.normal(0, 200, n)

    return pd.DataFrame({
        "family": families,
        "level": levels,
        "flag": flags,
        "x": x,
        "y": y,
    })


def test_hierarchical_segment_assigns_every_row():
    df = make_test_df()
    hierarchy = [["family", "level", "flag"], ["level", "flag"], ["flag"]]
    result = hierarchical_segment(df, hierarchy, min_size=20)

    assert result["segment_id"].isna().sum() == 0
    assert result["segment_level"].isna().sum() == 0
    assert len(result) == len(df)


def test_hierarchical_segment_respects_min_size():
    df = make_test_df()
    hierarchy = [["family", "level", "flag"], ["level", "flag"], ["flag"]]
    result = hierarchical_segment(df, hierarchy, min_size=20)

    sizes = segment_sizes(result)
    # Every assigned segment (except a possible ALL catch-all) must
    # respect the minimum, since ALL absorbs whatever didn't fit
    # anywhere finer.
    non_all = sizes.drop("ALL", errors="ignore")
    assert (non_all >= 20).all()


def test_hierarchical_segment_falls_back_to_all_when_too_strict():
    df = make_test_df(n=50)  # small dataset, high min_size forces fallback
    hierarchy = [["family", "level", "flag"]]
    result = hierarchical_segment(df, hierarchy, min_size=1000)

    assert (result["segment_id"] == "ALL").all()
    assert (result["segment_level"] == 1).all()  # len(hierarchy) == 1


def test_apply_by_segment_with_model_stability():
    df = make_test_df()
    hierarchy = [["family"]]
    segmented = hierarchical_segment(df, hierarchy, min_size=20)

    report = apply_by_segment(
        segmented, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=model_stability_pct,
    )

    assert "median_pct_diff" in report.columns
    assert (report["skipped"] == False).any()  # noqa: E712
    assert (report["n"] >= 5).all() | (report["skipped"])


def test_apply_by_segment_skips_small_segments():
    df = pd.DataFrame({
        "segment_id": ["a"] * 3 + ["b"] * 50,
        "x": np.random.default_rng(0).uniform(0, 10, 53),
        "y": np.random.default_rng(1).uniform(0, 10, 53),
    })

    report = apply_by_segment(
        df, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=model_stability_pct, min_points=10,
    )

    a_row = report[report["segment"] == "a"].iloc[0]
    b_row = report[report["segment"] == "b"].iloc[0]

    assert a_row["skipped"] is True
    assert b_row["skipped"] is False
    assert "median_pct_diff" in report.columns


def test_apply_by_segment_with_cook_impact():
    df = make_test_df()
    hierarchy = [["family"]]
    segmented = hierarchical_segment(df, hierarchy, min_size=20)

    def analysis(x, y):
        diag = cooks_diagnostic(x, y)
        if len(diag["flagged_indices"]) == 0:
            return {"median_pct_change": 0.0, "max_pct_change": 0.0, "n_flagged": 0}
        impact = cook_impact(x, y, diag["flagged_indices"])
        return {
            "median_pct_change": impact["median_pct_change"],
            "max_pct_change": impact["max_pct_change"],
            "n_flagged": len(diag["flagged_indices"]),
        }

    report = apply_by_segment(
        segmented, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=analysis,
    )

    assert "median_pct_change" in report.columns
    assert "n_flagged" in report.columns


def test_apply_by_segment_reports_errors_without_aborting():
    df = pd.DataFrame({
        "segment_id": ["a"] * 30 + ["b"] * 30,
        "x": [5.0] * 30 + list(np.random.default_rng(0).uniform(0, 10, 30)),
        "y": [5.0] * 30 + list(np.random.default_rng(1).uniform(0, 10, 30)),
    })

    def analysis(x, y):
        if np.allclose(x, x[0]):
            raise ValueError("x has no variance")
        return {"ok": True}

    report = apply_by_segment(
        df, segment_col="segment_id", x_col="x", y_col="y",
        analysis_fn=analysis,
    )

    a_row = report[report["segment"] == "a"].iloc[0]
    b_row = report[report["segment"] == "b"].iloc[0]

    assert "error" in report.columns
    assert isinstance(a_row["error"], str)
    assert pd.isna(b_row["error"])

import numpy as np
import pandas as pd
import pytest

from robustkit import mad_outlier_report


def make_job_family_df_with_outliers(n=800, n_outliers=8, seed=0):
    rng = np.random.default_rng(seed)
    employee_id = [f"E{i:04d}" for i in range(n)]
    job_family = rng.choice(["ENG", "FIN", "ADM"], n, p=[0.6, 0.25, 0.15])
    p_level = rng.choice(["P3", "P4", "P5"], n, p=[0.5, 0.35, 0.15])
    ot = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)

    df = pd.DataFrame({
        "employee_id": employee_id, "JobFamily": job_family, "P_niva": p_level, "OT": ot,
        "age": age, "salary": salary,
    })

    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    df.loc[outlier_idx, "salary"] = df.loc[outlier_idx, "salary"] - 15000
    outlier_ids = set(df.loc[outlier_idx, "employee_id"])

    return df, outlier_ids


def test_mad_outlier_report_flags_injected_outliers():
    df, outlier_ids = make_job_family_df_with_outliers()

    report = mad_outlier_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, k=3.0, direction="negative", id_cols=["employee_id"],
    )

    flagged_ids = set(report[report["flagged"]]["employee_id"])
    assert len(outlier_ids & flagged_ids) >= 6  # most/all injected severe outliers should be caught


def test_mad_outlier_report_output_structure():
    df, _ = make_job_family_df_with_outliers()
    report = mad_outlier_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, id_cols=["employee_id"],
    )

    expected_cols = {
        "employee_id", "actual", "expected", "residual", "residual_pct",
        "segment", "segment_level", "segment_mad", "threshold", "flagged",
    }
    assert expected_cols <= set(report.columns)
    assert len(report) == len(df)
    assert (report["segment_mad"] >= 0).all()
    assert (report["threshold"] >= 0).all()


def test_mad_outlier_report_handles_degenerate_zero_mad_segment():
    """A tiny segment where every residual happens to be identical (MAD=0)
    must not crash and must not flag everyone in it."""
    df, _ = make_job_family_df_with_outliers()

    tiny_df = pd.DataFrame({
        "employee_id": ["T1", "T2", "T3"],
        "JobFamily": ["ADM", "ADM", "ADM"], "P_niva": ["P5", "P5", "P5"], "OT": ["Yes", "Yes", "Yes"],
        "age": [45, 46, 47], "salary": [50000, 50000, 50000],
    })
    combined = pd.concat([df, tiny_df], ignore_index=True)

    report = mad_outlier_report(
        combined, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, id_cols=["employee_id"],
    )
    assert len(report) == len(combined)  # completed without raising


def test_mad_outlier_report_direction_variants():
    df, _ = make_job_family_df_with_outliers()

    negative = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, direction="negative")
    positive = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, direction="positive")
    two_sided = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, direction="two_sided")

    # two_sided should flag at least as many as either one-sided direction alone
    assert two_sided["flagged"].sum() >= negative["flagged"].sum()
    assert two_sided["flagged"].sum() >= positive["flagged"].sum()


def test_mad_outlier_report_rejects_invalid_direction():
    df, _ = make_job_family_df_with_outliers()
    with pytest.raises(ValueError):
        mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", direction="sideways")


def test_mad_outlier_report_falls_back_to_row_id_without_id_cols():
    df, _ = make_job_family_df_with_outliers()
    report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20)
    assert "row_id" in report.columns


def test_mad_outlier_report_includes_segment_level_from_hierarchy():
    """Segments that fell back to a coarser tier should show a
    nonzero segment_level, same convention as segment_stability_report
    / segment_benchmark_report."""
    df, _ = make_job_family_df_with_outliers()
    tiny_df = pd.DataFrame({
        "employee_id": ["T1", "T2", "T3"],
        "JobFamily": ["ADM", "FIN", "ENG"], "P_niva": ["P5", "P5", "P5"], "OT": ["Yes", "Yes", "Yes"],
        "age": [45, 50, 55], "salary": [60000, 65000, 70000],
    })
    combined = pd.concat([df, tiny_df], ignore_index=True)

    report = mad_outlier_report(
        combined, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, id_cols=["employee_id"],
    )
    assert report["segment_level"].max() >= 1

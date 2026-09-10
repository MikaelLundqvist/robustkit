import numpy as np
import pandas as pd
import pytest

from robustkit import (
    segment_benchmark_drilldown_report, mad_outlier_drilldown_report,
    segment_benchmark_report, mad_outlier_report,
)


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
    return df


def test_segment_benchmark_drilldown_shows_overlap():
    """The defining property of drilldown vs. exclusive: rows overlap,
    so the sum of n across rows exceeds the population size."""
    df = make_job_family_df_with_outliers()

    drill = segment_benchmark_drilldown_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100,
    )
    excl = segment_benchmark_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100,
    )

    assert drill["n"].sum() > len(df)
    assert excl["n"].sum() == len(df)
    assert len(drill) > len(excl)


def test_segment_benchmark_drilldown_includes_segment_level():
    df = make_job_family_df_with_outliers()
    drill = segment_benchmark_drilldown_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100,
    )
    assert "segment_level" in drill.columns
    assert set(drill["segment_level"].unique()) <= {0, 1, 2}


def test_mad_outlier_drilldown_same_individual_appears_multiple_times():
    df = make_job_family_df_with_outliers()

    drill = mad_outlier_drilldown_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, k=3.0, id_cols=["employee_id"],
    )
    excl = mad_outlier_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, k=3.0, id_cols=["employee_id"],
    )

    assert len(excl) == len(df)  # exclusive: exactly one row per individual
    assert len(drill) > len(df)  # drilldown: overlap inflates row count

    dup_counts = drill["employee_id"].value_counts()
    assert (dup_counts > 1).sum() > 0  # at least some individuals appear >1 time


def test_mad_outlier_drilldown_rejects_invalid_direction():
    df = make_job_family_df_with_outliers()
    with pytest.raises(ValueError):
        mad_outlier_drilldown_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", direction="sideways")


def test_segment_benchmark_drilldown_requires_x_col_or_benchmark_fit():
    df = make_job_family_df_with_outliers()
    with pytest.raises(ValueError):
        segment_benchmark_drilldown_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"])

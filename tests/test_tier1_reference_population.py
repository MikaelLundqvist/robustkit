import numpy as np
import pandas as pd

from robustkit import (
    segment_stability_drilldown_report, segment_stability_report,
    mad_outlier_drilldown_report, outlier_drilldown_summary,
)


def make_job_family_df_with_persistent_outliers(n=800, n_outliers=5, seed=0):
    rng = np.random.default_rng(seed)
    employee_id = [f"E{i:04d}" for i in range(n)]
    job_family = rng.choice(["ENG", "FIN", "ADM"], n, p=[0.6, 0.25, 0.15])
    p_level = rng.choice(["P3", "P4", "P5"], n, p=[0.5, 0.35, 0.15])
    ot = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)

    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    salary[outlier_idx] -= 20000
    outlier_ids = {employee_id[i] for i in outlier_idx}

    df = pd.DataFrame({
        "employee_id": employee_id, "JobFamily": job_family, "P_niva": p_level, "OT": ot,
        "age": age, "salary": salary,
    })
    return df, outlier_ids


def test_segment_stability_drilldown_shows_overlap():
    df, _ = make_job_family_df_with_persistent_outliers()

    drill = segment_stability_drilldown_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)
    excl = segment_stability_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    assert len(drill) > len(excl)
    assert set(drill["segment_level"].unique()) <= {0, 1, 2}


def test_segment_stability_drilldown_matches_apply_by_segment_row_shape():
    df, _ = make_job_family_df_with_persistent_outliers()
    drill = segment_stability_drilldown_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    assert {"segment", "segment_level", "n", "skipped"} <= set(drill.columns)


def test_outlier_drilldown_summary_identifies_persistent_outliers():
    df, outlier_ids = make_job_family_df_with_persistent_outliers()

    drill = mad_outlier_drilldown_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, k=3.0, id_cols=["employee_id"],
    )
    summary = outlier_drilldown_summary(drill, id_col="employee_id")

    assert set(summary.columns) == {"employee_id", "outlier_levels", "levels"}
    assert (summary["outlier_levels"] >= 1).all()

    max_levels = summary["outlier_levels"].max()
    top_survivors = set(summary[summary["outlier_levels"] == max_levels]["employee_id"])
    assert len(outlier_ids & top_survivors) >= 2  # at least some injected outliers should be top survivors


def test_outlier_drilldown_summary_empty_when_nothing_flagged():
    empty_report = pd.DataFrame({"employee_id": ["A", "B"], "flagged": [False, False], "segment_level": [0, 0]})
    summary = outlier_drilldown_summary(empty_report, id_col="employee_id")
    assert len(summary) == 0
    assert set(summary.columns) == {"employee_id", "outlier_levels", "levels"}


def test_outlier_drilldown_summary_sorted_descending_by_levels():
    df, _ = make_job_family_df_with_persistent_outliers()
    drill = mad_outlier_drilldown_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, k=3.0, id_cols=["employee_id"],
    )
    summary = outlier_drilldown_summary(drill, id_col="employee_id")
    assert (summary["outlier_levels"].diff().dropna() <= 0).all()

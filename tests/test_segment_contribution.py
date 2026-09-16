import numpy as np
import pandas as pd
import pytest

from robustkit import segment_contribution_report


def make_contribution_scenario(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "FIN", "ITS", "ADM"], n, p=[0.35, 0.25, 0.25, 0.15])
    p_level = rng.choice(["P2", "P3", "P4"], n, p=[0.4, 0.4, 0.2])
    ot = rng.choice(["Yes", "No"], n, p=[0.5, 0.5])
    age = rng.uniform(25, 60, n)

    salary = 30000 + 400 * age
    eng_p2_yes_mask = (job_family == "ENG") & (p_level == "P2") & (ot == "Yes")
    salary = salary + np.where(eng_p2_yes_mask, 1200, 0)
    its_p3_no_mask = (job_family == "ITS") & (p_level == "P3") & (ot == "No")
    salary = salary + np.where(its_p3_no_mask, -4800, 0)
    salary = salary + rng.normal(0, 800, n)

    return pd.DataFrame({"JobFamily": job_family, "P_niva": p_level, "OT": ot, "age": age, "salary": salary})


def test_segment_contribution_report_structure():
    df = make_contribution_scenario()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100)
    expected_cols = {
        "segment", "segment_level", "n", "observed_median", "expected_median",
        "difference", "ci_lower", "ci_upper", "parent_segment", "contribution",
    }
    assert expected_cols <= set(report.columns)


def test_segment_contribution_report_positive_contribution_direction():
    df = make_contribution_scenario()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100)
    eng_row = report[report["segment"] == "ENG_P2_Yes"].iloc[0]
    assert eng_row["contribution"] > 500
    assert eng_row["parent_segment"] == "P2_Yes"


def test_segment_contribution_report_negative_contribution_direction():
    df = make_contribution_scenario()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100)
    its_row = report[report["segment"] == "ITS_P3_No"].iloc[0]
    assert its_row["contribution"] < -2000
    assert its_row["parent_segment"] == "P3_No"


def test_segment_contribution_report_arithmetic_is_exact():
    """contribution must equal exactly child_difference - parent_difference."""
    df = make_contribution_scenario()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100)

    child = report[report["segment"] == "ENG_P2_Yes"].iloc[0]
    parent = report[report["segment"] == "P2_Yes"].iloc[0]
    assert abs(child["contribution"] - (child["difference"] - parent["difference"])) < 1e-6


def test_segment_contribution_report_coarsest_level_has_no_parent():
    df = make_contribution_scenario()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100)
    coarsest = report[report["segment_level"] == report["segment_level"].max()]
    assert coarsest["contribution"].isna().all()
    assert coarsest["parent_segment"].isna().all()


def test_segment_contribution_report_requires_x_col_or_benchmark_fit():
    df = make_contribution_scenario()
    with pytest.raises(ValueError):
        segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"])

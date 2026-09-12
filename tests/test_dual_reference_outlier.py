import numpy as np
import pandas as pd
import pytest

from robustkit import dual_reference_outlier_report


def make_four_type_scenario(seed=0):
    """
    A scenario engineered to produce genuine examples of all four
    reference_type classifications:

      - Majority (400, baseline): a few severe individual drops -> Type A
      - UnderPaid (50, MINORITY, offset -4000, normal noise): whole
        segment collectively low relative to the majority-dominated
        global benchmark, but typical within itself -> Type C
      - TightTeam (50, baseline level, but tight internal noise): a
        few modest dips that stand out locally but not globally -> Type B
    """
    rng = np.random.default_rng(seed)

    n_maj = 400
    age_maj = rng.uniform(25, 60, n_maj)
    salary_maj = 30000 + 400 * age_maj + rng.normal(0, 800, n_maj)
    severe_idx = rng.choice(n_maj, size=4, replace=False)
    salary_maj[severe_idx] -= 20000

    n_under = 50
    age_under = rng.uniform(25, 60, n_under)
    salary_under = 30000 + 400 * age_under - 4000 + rng.normal(0, 800, n_under)

    n_tight = 50
    age_tight = rng.uniform(25, 60, n_tight)
    salary_tight = 30000 + 400 * age_tight + rng.normal(0, 150, n_tight)
    tight_dip_idx = rng.choice(n_tight, size=3, replace=False)
    salary_tight[tight_dip_idx] -= 2500

    employee_id = (
        [f"MAJ_{i}" for i in range(n_maj)]
        + [f"UNDER_{i}" for i in range(n_under)]
        + [f"TIGHT_{i}" for i in range(n_tight)]
    )
    segment_tag = ["Majority"] * n_maj + ["UnderPaid"] * n_under + ["TightTeam"] * n_tight
    age = np.concatenate([age_maj, age_under, age_tight])
    salary = np.concatenate([salary_maj, salary_under, salary_tight])

    df = pd.DataFrame({"employee_id": employee_id, "segment_tag": segment_tag, "age": age, "salary": salary})

    ids = {
        "type_a_candidates": {f"MAJ_{i}" for i in severe_idx},
        "type_c_candidates": {f"UNDER_{i}" for i in range(n_under)},
        "type_b_candidates": {f"TIGHT_{i}" for i in tight_dip_idx},
    }
    return df, ids


def test_dual_reference_outlier_report_structure():
    df, _ = make_four_type_scenario()
    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20, k=3.0, id_cols=["employee_id"],
    )
    expected_cols = {
        "employee_id", "actual", "local_expected", "local_residual", "local_flagged",
        "global_expected", "global_residual", "global_flagged", "segment", "segment_level", "reference_type",
    }
    assert expected_cols <= set(report.columns)
    assert set(report["reference_type"].unique()) <= {"A", "B", "C", "D"}


def test_dual_reference_outlier_report_produces_type_a_for_severe_individual_drops():
    df, ids = make_four_type_scenario()
    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20, k=3.0, id_cols=["employee_id"],
    )
    subset = report[report["employee_id"].isin(ids["type_a_candidates"])]
    assert (subset["reference_type"] == "A").mean() >= 0.75


def test_dual_reference_outlier_report_produces_type_c_for_minority_segment_shift():
    """A segment collectively shifted below a majority-dominated global
    benchmark, but typical within itself, should mostly classify as C --
    NOT be silently absorbed the way per-segment MAD centering would."""
    df, ids = make_four_type_scenario()
    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20, k=3.0, id_cols=["employee_id"],
    )
    subset = report[report["employee_id"].isin(ids["type_c_candidates"])]
    assert (subset["reference_type"] == "C").mean() > 0.7


def test_dual_reference_outlier_report_produces_type_b_for_tight_segment_dips():
    df, ids = make_four_type_scenario()
    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20, k=3.0, id_cols=["employee_id"],
    )
    subset = report[report["employee_id"].isin(ids["type_b_candidates"])]
    assert (subset["reference_type"] == "B").mean() >= 0.6


def test_dual_reference_outlier_report_majority_is_mostly_type_d():
    df, ids = make_four_type_scenario()
    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20, k=3.0, id_cols=["employee_id"],
    )
    majority_normal = report[
        report["employee_id"].str.startswith("MAJ_") & ~report["employee_id"].isin(ids["type_a_candidates"])
    ]
    assert (majority_normal["reference_type"] == "D").mean() > 0.9


def test_dual_reference_outlier_report_rejects_invalid_direction():
    df, _ = make_four_type_scenario()
    with pytest.raises(ValueError):
        dual_reference_outlier_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag"], direction="sideways")


def test_dual_reference_outlier_report_falls_back_to_row_id_without_id_cols():
    df, _ = make_four_type_scenario()
    report = dual_reference_outlier_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag"], min_size=20)
    assert "row_id" in report.columns

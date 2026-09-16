import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import export_huber_iqr_pdf, dual_reference_outlier_report, dual_reference_outlier_drilldown_report


def make_job_family_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "FIN", "ADM"], n, p=[0.6, 0.25, 0.15])
    p_level = rng.choice(["P3", "P4", "P5"], n, p=[0.5, 0.35, 0.15])
    ot = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    return pd.DataFrame({"JobFamily": job_family, "P_niva": p_level, "OT": ot, "age": age, "salary": salary})


def make_four_type_scenario_multilevel(seed=0):
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
        [f"MAJ_{i}" for i in range(n_maj)] + [f"UNDER_{i}" for i in range(n_under)] + [f"TIGHT_{i}" for i in range(n_tight)]
    )
    segment_tag = ["Majority"] * n_maj + ["UnderPaid"] * n_under + ["TightTeam"] * n_tight
    region = rng.choice(["North", "South"], n_maj + n_under + n_tight)
    age = np.concatenate([age_maj, age_under, age_tight])
    salary = np.concatenate([salary_maj, salary_under, salary_tight])

    df = pd.DataFrame({"employee_id": employee_id, "segment_tag": segment_tag, "region": region, "age": age, "salary": salary})
    ids = {
        "type_a_candidates": {f"MAJ_{i}" for i in severe_idx},
        "type_c_candidates": {f"UNDER_{i}" for i in range(n_under)},
    }
    return df, ids


# ---------------------------------------------------------------------------
# export_huber_iqr_pdf: mode="exclusive"/"drilldown"
# ---------------------------------------------------------------------------

def test_export_huber_iqr_pdf_mode_exclusive_is_default():
    import inspect
    sig = inspect.signature(export_huber_iqr_pdf)
    assert sig.parameters["mode"].default == "exclusive"


def test_export_huber_iqr_pdf_rejects_invalid_mode():
    df = make_job_family_df()
    with pytest.raises(ValueError):
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path="x.pdf", mode="bogus")


def test_export_huber_iqr_pdf_drilldown_has_more_pages_than_exclusive():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path_excl = os.path.join(tmp, "excl.pdf")
        path_drill = os.path.join(tmp, "drill.pdf")

        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], path=path_excl, min_size=20, min_points_to_plot=5, mode="exclusive")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], path=path_drill, min_size=20, min_points_to_plot=5, mode="drilldown")

        n_excl = len(pypdf.PdfReader(path_excl).pages)
        n_drill = len(pypdf.PdfReader(path_drill).pages)
        assert n_drill > n_excl


def test_export_huber_iqr_pdf_drilldown_title_fn_receives_segment_level():
    df = make_job_family_df()
    seen_levels = set()

    def title_fn(segment_name, info):
        seen_levels.add(info["segment_level"])
        return segment_name

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "drill.pdf")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], path=path, min_size=20, min_points_to_plot=5, mode="drilldown", title_fn=title_fn)

    assert seen_levels <= {0, 1, 2}
    assert len(seen_levels) > 1  # drilldown should span multiple levels


# ---------------------------------------------------------------------------
# dual_reference_outlier_drilldown_report
# ---------------------------------------------------------------------------

def test_dual_reference_outlier_drilldown_shows_overlap():
    df, _ = make_four_type_scenario_multilevel()
    drill = dual_reference_outlier_drilldown_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag", "region"], min_size=20, k=3.0, id_cols=["employee_id"])
    excl = dual_reference_outlier_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag", "region"], min_size=20, k=3.0, id_cols=["employee_id"])

    assert len(drill) > len(excl)
    dup_counts = drill["employee_id"].value_counts()
    assert (dup_counts > 1).sum() > 0


def test_dual_reference_outlier_drilldown_structure():
    df, _ = make_four_type_scenario_multilevel()
    drill = dual_reference_outlier_drilldown_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag", "region"], min_size=20, k=3.0, id_cols=["employee_id"])
    expected_cols = {
        "employee_id", "actual", "local_expected", "local_residual", "local_flagged",
        "global_expected", "global_residual", "global_flagged", "segment", "segment_level", "reference_type",
    }
    assert expected_cols <= set(drill.columns)
    assert set(drill["reference_type"].unique()) <= {"A", "B", "C", "D"}


def test_dual_reference_outlier_drilldown_produces_type_a_and_type_c():
    df, ids = make_four_type_scenario_multilevel()
    drill = dual_reference_outlier_drilldown_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag", "region"], min_size=20, k=3.0, id_cols=["employee_id"])

    severe = drill[drill["employee_id"].isin(ids["type_a_candidates"])]
    assert (severe["reference_type"] == "A").mean() > 0.5

    under = drill[drill["employee_id"].isin(ids["type_c_candidates"])]
    assert (under["reference_type"] == "C").mean() > 0.3


def test_dual_reference_outlier_drilldown_rejects_invalid_direction():
    df, _ = make_four_type_scenario_multilevel()
    with pytest.raises(ValueError):
        dual_reference_outlier_drilldown_report(df, y_col="salary", x_col="age", segment_cols=["segment_tag"], direction="sideways")

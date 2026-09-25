import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import mad_outlier_report, mad_outlier_drilldown_report, export_outlier_pdf, hierarchical_segment
from robustkit.core.trend import fit_huber_trend, predict_trend
from robustkit.benchmark.global_model import fit_huber_benchmark, benchmark_predict


def make_df_with_outliers(n=400, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n)
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    outlier_idx = rng.choice(n, size=10, replace=False)
    salary[outlier_idx] -= 8000
    return pd.DataFrame({"JobFamily": job_family, "age": age, "salary": salary, "emp_id": [f"E{i}" for i in range(n)]}), set(f"E{i}" for i in outlier_idx)


# ---------------------------------------------------------------------------
# mad_outlier_report: reference="benchmark_curve"
# ---------------------------------------------------------------------------

def test_mad_outlier_report_accepts_benchmark_curve():
    df, true_outliers = make_df_with_outliers()
    report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                 min_size=20, k=3.0, id_cols=["emp_id"], reference="benchmark_curve")
    flagged = set(report[report["flagged"]]["emp_id"])
    assert len(flagged & true_outliers) == len(true_outliers)


def test_mad_outlier_report_benchmark_curve_requires_x_col():
    df, _ = make_df_with_outliers()
    with pytest.raises(ValueError):
        mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], reference="benchmark_curve")


def test_mad_outlier_report_rejects_invalid_reference_still():
    df, _ = make_df_with_outliers()
    with pytest.raises(ValueError):
        mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", reference="bogus")


# ---------------------------------------------------------------------------
# mad_outlier_drilldown_report: reference="benchmark_curve", consistency
# ---------------------------------------------------------------------------

def test_mad_outlier_drilldown_report_benchmark_curve_matches_exclusive_at_level_0():
    df, _ = make_df_with_outliers()
    excl = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                               min_size=20, k=3.0, id_cols=["emp_id"], reference="benchmark_curve")
    drill = mad_outlier_drilldown_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                          min_size=20, k=3.0, id_cols=["emp_id"], reference="benchmark_curve")
    drill_level0 = drill[drill["segment_level"] == 0]

    flagged_excl = set(excl[excl["flagged"]]["emp_id"])
    flagged_drill0 = set(drill_level0[drill_level0["flagged"]]["emp_id"])
    assert flagged_excl == flagged_drill0


# ---------------------------------------------------------------------------
# export_outlier_pdf: reference="benchmark_curve", all combinations,
# and the critical consistency check
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_accepts_benchmark_curve_reference():
    df, _ = make_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.pdf")
        result = export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                     path=path, min_size=20, min_points_to_plot=5, reference="benchmark_curve")
        assert os.path.exists(result)


def test_export_outlier_pdf_all_six_mode_reference_combinations_run():
    df, _ = make_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("benchmark", "local"):
            for reference in ("model", "curve", "benchmark_curve"):
                path = os.path.join(tmp, f"{mode}_{reference}.pdf")
                result = export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                             path=path, min_size=20, min_points_to_plot=5,
                                             mode=mode, reference=reference)
                assert os.path.exists(result)


def test_export_outlier_pdf_benchmark_curve_flagging_matches_mad_outlier_report_exactly():
    """The critical consistency check, same pattern as Tier 2's for
    reference='curve': export_outlier_pdf's internal flagging with
    reference='benchmark_curve' must exactly match
    mad_outlier_report(reference='benchmark_curve')."""
    df, _ = make_df_with_outliers()
    numeric_report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                         min_size=20, k=3.0, id_cols=["emp_id"], reference="benchmark_curve")
    flagged_from_report = set(numeric_report[numeric_report["flagged"]]["emp_id"])

    benchmark_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=2)
    segmented = hierarchical_segment(df, [["JobFamily"]], min_size=20)
    flagged_manual = set()
    for seg_id, group in segmented.groupby("segment_id", observed=True):
        x = group["age"].to_numpy(dtype=float)
        y = group["salary"].to_numpy(dtype=float)
        raw_expected = np.asarray(benchmark_predict(benchmark_fit, group, x_col="age"), dtype=float)
        curve_fit = fit_huber_trend(x, raw_expected, degree=2)
        expected = predict_trend(curve_fit, x)
        residual = y - expected
        median_residual = float(np.median(residual))
        mad_val = float(1.4826 * np.median(np.abs(residual - median_residual)))
        safe_mad = mad_val if mad_val > 0 else np.inf
        z = (residual - median_residual) / safe_mad
        flagged_manual.update(group["emp_id"].to_numpy()[z < -3.0])

    assert flagged_from_report == flagged_manual


def test_export_outlier_pdf_benchmark_mode_with_benchmark_curve_reference_is_fully_consistent():
    """mode='benchmark' + reference='benchmark_curve' should show the
    curve and the flagging reference as the SAME thing -- no '(not
    the curve shown)' caveat, the new equivalent of the mode='local' +
    reference='curve' guarantee from Tier 2."""
    pytest.importorskip("pypdf")
    import pypdf
    df, _ = make_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "consistent.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path,
                            min_size=20, min_points_to_plot=5, mode="benchmark", reference="benchmark_curve")
        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "not the curve shown" not in text


def test_export_outlier_pdf_mode_local_with_benchmark_curve_reference_flags_the_gap():
    """mode='local' + reference='benchmark_curve' draws one thing but
    flags against another -- the info box must say so."""
    pytest.importorskip("pypdf")
    import pypdf
    df, _ = make_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mismatched.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path,
                            min_size=20, min_points_to_plot=5, mode="local", reference="benchmark_curve")
        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "not the curve shown" in text

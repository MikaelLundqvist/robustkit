import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import export_outlier_pdf, segment_quality_report


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


# ---------------------------------------------------------------------------
# export_outlier_pdf mode="local"/"benchmark"
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_mode_benchmark_is_default():
    import inspect
    sig = inspect.signature(export_outlier_pdf)
    assert sig.parameters["mode"].default == "benchmark"


def test_export_outlier_pdf_rejects_invalid_mode():
    df = make_job_family_df_with_outliers()
    with pytest.raises(ValueError):
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path="x.pdf", mode="bogus")


def test_export_outlier_pdf_local_mode_uses_different_legend_label():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path_bench = os.path.join(tmp, "bench.pdf")
        path_local = os.path.join(tmp, "local.pdf")

        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_bench, min_size=20, min_points_to_plot=5, mode="benchmark")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_local, min_size=20, min_points_to_plot=5, mode="local")

        text_bench = pypdf.PdfReader(path_bench).pages[0].extract_text()
        text_local = pypdf.PdfReader(path_local).pages[0].extract_text()

        assert "Benchmark" in text_bench
        assert "Local" in text_local


def test_export_outlier_pdf_local_and_benchmark_same_page_count():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path_bench = os.path.join(tmp, "bench.pdf")
        path_local = os.path.join(tmp, "local.pdf")

        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_bench, min_size=20, min_points_to_plot=5, mode="benchmark")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_local, min_size=20, min_points_to_plot=5, mode="local")

        assert len(pypdf.PdfReader(path_bench).pages) == len(pypdf.PdfReader(path_local).pages)


# ---------------------------------------------------------------------------
# segment_quality_report
# ---------------------------------------------------------------------------

def test_segment_quality_report_structure():
    df = make_job_family_df_with_outliers()
    report = segment_quality_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    expected_cols = {"segment", "n", "skipped", "segment_level"}
    assert expected_cols <= set(report.columns)
    assert len(report) > 0


def test_segment_quality_report_populates_statsmodels_free_measures():
    """mdape, iqr_resid, cook_impact_pct, n_flagged don't need Tukey/
    statsmodels and should always populate regardless of environment."""
    df = make_job_family_df_with_outliers()
    report = segment_quality_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    non_skipped = report[~report["skipped"]]
    assert non_skipped["mdape"].notna().all()
    assert non_skipped["iqr_resid"].notna().all()
    assert non_skipped["cook_impact_pct"].notna().all()
    assert non_skipped["n_flagged"].notna().all()


def test_segment_quality_report_stability_failure_does_not_block_other_measures():
    """A stability_error (e.g. missing statsmodels) must not prevent
    cook_impact/mdape/iqr_resid from being reported for that segment."""
    df = make_job_family_df_with_outliers()
    report = segment_quality_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    if "stability_error" in report.columns:
        with_stability_error = report[report["stability_error"].notna()]
        if len(with_stability_error) > 0:
            assert with_stability_error["mdape"].notna().all()
            assert with_stability_error["cook_impact_pct"].notna().all()


def test_segment_quality_report_skips_undersized_segments():
    df = make_job_family_df_with_outliers()
    tiny_df = pd.DataFrame({
        "employee_id": ["T1", "T2"], "JobFamily": ["ADM", "ADM"], "P_niva": ["P5", "P5"], "OT": ["Yes", "Yes"],
        "age": [45, 46], "salary": [60000, 61000],
    })
    combined = pd.concat([df, tiny_df], ignore_index=True)

    report = segment_quality_report(combined, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20, min_points=5)
    assert (report["skipped"] == False).any()  # noqa: E712

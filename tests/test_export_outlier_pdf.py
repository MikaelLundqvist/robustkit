import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import export_outlier_pdf, mad_outlier_report


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


def test_export_outlier_pdf_creates_file():
    df = make_job_family_df_with_outliers()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "outliers.pdf")
        result_path = export_outlier_pdf(
            df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path,
            min_size=20, k=3.0, id_cols=["employee_id"], min_points_to_plot=5,
        )
        assert result_path == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # not a trivially empty file


def test_export_outlier_pdf_page_count_matches_qualifying_segments():
    pytest.importorskip("pypdf")  # only needed to verify page count in this test
    import pypdf

    df = make_job_family_df_with_outliers()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "outliers.pdf")
        export_outlier_pdf(
            df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path,
            min_size=20, k=3.0, id_cols=["employee_id"], min_points_to_plot=5,
        )

        actual_pages = len(pypdf.PdfReader(path).pages)

    report = mad_outlier_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
        min_size=20, id_cols=["employee_id"],
    )
    expected_pages = int((report.groupby("segment").size() >= 5).sum())

    assert actual_pages == expected_pages


def test_export_outlier_pdf_min_points_to_plot_reduces_page_count():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df_with_outliers()

    with tempfile.TemporaryDirectory() as tmp:
        path_low = os.path.join(tmp, "low_threshold.pdf")
        path_high = os.path.join(tmp, "high_threshold.pdf")

        export_outlier_pdf(
            df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_low,
            min_size=20, min_points_to_plot=5,
        )
        export_outlier_pdf(
            df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path_high,
            min_size=20, min_points_to_plot=40,
        )

        pages_low = len(pypdf.PdfReader(path_low).pages)
        pages_high = len(pypdf.PdfReader(path_high).pages)

    assert pages_high < pages_low


def test_export_outlier_pdf_works_with_custom_benchmark_model():
    df = make_job_family_df_with_outliers()

    class SimpleModel:
        def __init__(self, df):
            X = np.column_stack([np.ones(len(df)), df["age"].to_numpy(dtype=float)])
            self.beta, *_ = np.linalg.lstsq(X, df["salary"].to_numpy(dtype=float), rcond=None)

        def predict(self, df):
            X = np.column_stack([np.ones(len(df)), df["age"].to_numpy(dtype=float)])
            return X @ self.beta

    model = SimpleModel(df)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "custom_model.pdf")
        export_outlier_pdf(
            df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", path=path,
            benchmark_fit=model, min_size=20, min_points_to_plot=5,
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

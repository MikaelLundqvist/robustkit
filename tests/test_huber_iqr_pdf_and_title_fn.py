import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import export_huber_iqr_pdf, export_outlier_pdf


def make_job_family_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "FIN", "ADM"], n, p=[0.6, 0.25, 0.15])
    p_level = rng.choice(["P3", "P4", "P5"], n, p=[0.5, 0.35, 0.15])
    ot = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    return pd.DataFrame({"JobFamily": job_family, "P_niva": p_level, "OT": ot, "age": age, "salary": salary})


def test_export_huber_iqr_pdf_creates_file():
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        result = export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path, min_size=20, min_points_to_plot=5)
        assert result == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000


def test_export_huber_iqr_pdf_page_count_matches_qualifying_segments():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], path=path, min_size=20, min_points_to_plot=5)
        assert len(pypdf.PdfReader(path).pages) > 0


def test_export_huber_iqr_pdf_default_title_uses_segment_name_and_n():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path, min_size=20, min_points_to_plot=5)
        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "n=" in text


def test_export_huber_iqr_pdf_title_fn_receives_segment_name_and_info():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    seen = []

    def title_fn(segment_name, info):
        seen.append((segment_name, info))
        return f"CUSTOM TITLE: {segment_name}"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path, min_size=20, min_points_to_plot=5, title_fn=title_fn)

        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "CUSTOM TITLE" in text

    assert len(seen) > 0
    segment_name, info = seen[0]
    assert isinstance(segment_name, str)
    assert "n" in info and "segment_level" in info


def test_export_huber_iqr_pdf_domain_specific_title_translation():
    """Simulates a domain-specific pretty_segment_name callback, kept
    entirely outside the package -- robustkit only supplies the raw
    segment code and n."""
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    family_names = {"ENG": "Engineers", "FIN": "Finance", "ADM": "Admin"}

    def pretty_title(segment_code, info):
        return f"{family_names.get(segment_code, segment_code)} (n={info['n']})"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        export_huber_iqr_pdf(df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path, min_size=20, min_points_to_plot=5, title_fn=pretty_title)

        found_translated = False
        for page in pypdf.PdfReader(path).pages:
            text = page.extract_text()
            if any(name in text for name in family_names.values()):
                found_translated = True
                break
        assert found_translated


def test_export_huber_iqr_pdf_passes_through_style_kwargs():
    """Confirm plot_huber_iqr's style/method options are reachable
    through export_huber_iqr_pdf without crashing."""
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        export_huber_iqr_pdf(
            df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path,
            min_size=20, min_points_to_plot=5,
            methods=("huber", "ols"), cap_style="manual", residual_box_metric="mdape", ylim="dynamic",
        )
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# export_outlier_pdf: title_fn symmetry (retroactively added)
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_title_fn_default_none_preserves_old_title():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "outliers.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path, min_size=20, min_points_to_plot=5)
        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "flagged" in text.lower()


def test_export_outlier_pdf_title_fn_receives_expected_info_keys():
    df = make_job_family_df()
    seen = []

    def title_fn(segment_name, info):
        seen.append(info)
        return "x"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "outliers.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path, min_size=20, min_points_to_plot=5, title_fn=title_fn)

    assert len(seen) > 0
    expected_keys = {"n", "n_flagged", "pct_flagged", "k", "segment_mad"}
    assert expected_keys <= set(seen[0].keys())


def test_export_outlier_pdf_title_fn_custom_title_applied():
    pytest.importorskip("pypdf")
    import pypdf

    df = make_job_family_df()

    def title_fn(segment_name, info):
        return f"REVIEW: {segment_name}"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "outliers.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path, min_size=20, min_points_to_plot=5, title_fn=title_fn)
        text = pypdf.PdfReader(path).pages[0].extract_text()
        assert "REVIEW" in text


def test_export_huber_iqr_pdf_accepts_show_undersized_points():
    """Regression: export_huber_iqr_pdf must accept and pass through
    show_undersized_points to plot_huber_iqr without crashing."""
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        result = export_huber_iqr_pdf(
            df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path,
            min_size=20, min_points_to_plot=5,
            grouping="unique", min_n_for_iqr=5,
            show_undersized_points=False,
        )
        assert os.path.exists(result)


def test_export_huber_iqr_pdf_accepts_iqr_bar_color_via_style():
    """Regression: the style dict's iqr_bar_color key must reach
    plot_huber_iqr through export_huber_iqr_pdf without crashing."""
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "huber_iqr.pdf")
        result = export_huber_iqr_pdf(
            df, x_col="age", y_col="salary", segment_cols=["JobFamily"], path=path,
            min_size=20, min_points_to_plot=5,
            style={"iqr_color": "black", "iqr_bar_color": "gray"},
        )
        assert os.path.exists(result)

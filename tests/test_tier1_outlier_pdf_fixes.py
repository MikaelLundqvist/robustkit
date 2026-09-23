import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import export_outlier_pdf, mad_outlier_report, hierarchical_segment
from robustkit.benchmark.global_model import fit_huber_benchmark, benchmark_predict


def make_multivariate_df(seed=0, n_per_group=100):
    rng = np.random.default_rng(seed)
    rows = []

    def add(fam, lvl, ot, n, s):
        r = np.random.default_rng(s)
        age = r.uniform(25, 60, n)
        family_bump = {"ENG": 3000, "ITS": 1000, "FIN": 0}[fam]
        level_bump = {"M2": 0, "M3": 4000}[lvl]
        ot_bump = 1500 if ot == "Yes" else 0
        salary = 30000 + 400 * age + family_bump + level_bump + ot_bump + r.normal(0, 500, n)
        for a, sal in zip(age, salary):
            rows.append({"JobFamily": fam, "P_niva": lvl, "OT": ot, "age": a, "salary": sal, "emp_id": f"E{len(rows)}"})

    add("ENG", "M2", "No", n_per_group, seed + 1)
    add("ITS", "M2", "No", n_per_group, seed + 2)
    add("FIN", "M2", "No", n_per_group, seed + 3)
    add("ENG", "M3", "Yes", n_per_group, seed + 4)
    add("ITS", "M3", "Yes", n_per_group, seed + 5)
    add("FIN", "M3", "No", n_per_group, seed + 6)
    add("ENG", "M3", "No", n_per_group, seed + 7)
    add("ITS", "M3", "No", n_per_group, seed + 8)
    return pd.DataFrame(rows)


def make_multivariate_benchmark(df):
    X = pd.get_dummies(df[["age", "JobFamily", "P_niva", "OT"]], columns=["JobFamily", "P_niva", "OT"], drop_first=True)
    from sklearn.linear_model import HuberRegressor
    model = HuberRegressor()
    model.fit(X, df["salary"])

    class Wrapper:
        def predict(self, sub_df):
            Xs = pd.get_dummies(sub_df[["age", "JobFamily", "P_niva", "OT"]], columns=["JobFamily", "P_niva", "OT"], drop_first=True)
            Xs = Xs.reindex(columns=X.columns, fill_value=0)
            return model.predict(Xs)

    return Wrapper()


def make_job_family_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n)
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    outlier_idx = rng.choice(n, size=10, replace=False)
    salary[outlier_idx] -= 8000
    return pd.DataFrame({"JobFamily": job_family, "age": age, "salary": salary, "emp_id": [f"E{i}" for i in range(n)]})


# ---------------------------------------------------------------------------
# Curve fix: Huber fit to benchmark predictions instead of raw connected line
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_benchmark_curve_is_smooth_for_multivariate_model():
    """The core fix: a Huber curve fit to (x, expected) should be far
    smoother than connecting raw expected values sorted by x, for a
    segment that mixes multiple values of a covariate the benchmark
    model actually uses."""
    from robustkit.core.trend import fit_huber_trend, predict_trend

    df = make_multivariate_df()
    wrapper = make_multivariate_benchmark(df)

    # A coarse, mixed-covariate slice -- OT="No" spans JobFamily and P_niva
    seg = df[df["OT"] == "No"]
    x = seg["age"].to_numpy(dtype=float)
    expected = np.asarray(benchmark_predict(wrapper, seg, x_col="age"), dtype=float)

    order = x.argsort()
    raw_jaggedness = np.mean(np.abs(np.diff(expected[order])))

    curve_fit = fit_huber_trend(x, expected, degree=2)
    grid = np.linspace(x.min(), x.max(), 200)
    smooth_curve = predict_trend(curve_fit, grid)
    smooth_jaggedness = np.mean(np.abs(np.diff(smooth_curve)))

    assert smooth_jaggedness < raw_jaggedness / 5, "Huber-fit curve should be dramatically smoother than raw connected points"


def test_export_outlier_pdf_benchmark_mode_runs_without_crashing_on_multivariate_model():
    df = make_multivariate_df()
    wrapper = make_multivariate_benchmark(df)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.pdf")
        result = export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
                                     path=path, benchmark_fit=wrapper, min_size=20, min_points_to_plot=5,
                                     segment_mode="drilldown")
        assert os.path.exists(result)


# ---------------------------------------------------------------------------
# Flagging consistency: PDF's internal flagging must exactly match
# mad_outlier_report's numbers (this was the whole point of the fix)
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_flagging_matches_mad_outlier_report_exactly():
    df = make_job_family_df()
    numeric_report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                         min_size=20, k=3.0, id_cols=["emp_id"])
    flagged_from_report = set(numeric_report[numeric_report["flagged"]]["emp_id"])

    benchmark_fit = fit_huber_benchmark(df["age"].to_numpy(), df["salary"].to_numpy(), degree=2)
    segmented = hierarchical_segment(df, [["JobFamily"]], min_size=20)

    flagged_manual = set()
    for seg_id, group in segmented.groupby("segment_id", observed=True):
        x = group["age"].to_numpy(dtype=float)
        y = group["salary"].to_numpy(dtype=float)
        expected = np.asarray(benchmark_predict(benchmark_fit, group, x_col="age"), dtype=float)
        residual = y - expected
        median_residual = float(np.median(residual))
        mad_val = float(1.4826 * np.median(np.abs(residual - median_residual)))
        safe_mad = mad_val if mad_val > 0 else np.inf
        z = (residual - median_residual) / safe_mad
        flagged_manual.update(group["emp_id"].to_numpy()[z < -3.0])

    assert flagged_from_report == flagged_manual


# ---------------------------------------------------------------------------
# segment_mode: exclusive vs. drilldown
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_segment_mode_exclusive_is_default():
    import inspect
    sig = inspect.signature(export_outlier_pdf)
    assert sig.parameters["segment_mode"].default == "exclusive"


def test_export_outlier_pdf_rejects_invalid_segment_mode():
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                path=os.path.join(tmp, "x.pdf"), segment_mode="bogus")


def test_export_outlier_pdf_drilldown_has_more_or_equal_pages_than_exclusive():
    pytest.importorskip("pypdf")
    import pypdf
    df = make_multivariate_df()
    with tempfile.TemporaryDirectory() as tmp:
        path_excl = os.path.join(tmp, "excl.pdf")
        path_drill = os.path.join(tmp, "drill.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
                            path=path_excl, min_size=20, min_points_to_plot=5, segment_mode="exclusive")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age",
                            path=path_drill, min_size=20, min_points_to_plot=5, segment_mode="drilldown")
        n_excl = len(pypdf.PdfReader(path_excl).pages)
        n_drill = len(pypdf.PdfReader(path_drill).pages)
        assert n_drill >= n_excl


def test_export_outlier_pdf_title_fn_receives_segment_level():
    df = make_job_family_df()
    seen_levels = set()

    def title_fn(name, info):
        seen_levels.add(info["segment_level"])
        return name

    with tempfile.TemporaryDirectory() as tmp:
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                            path=os.path.join(tmp, "out.pdf"), min_size=20, min_points_to_plot=5, title_fn=title_fn)
    assert seen_levels  # at least one level was seen


# ---------------------------------------------------------------------------
# On-chart clarification of curve vs. flagging reference
# ---------------------------------------------------------------------------

def test_export_outlier_pdf_info_box_states_curve_and_flagging_reference():
    pytest.importorskip("pypdf")
    import pypdf
    df = make_job_family_df()
    with tempfile.TemporaryDirectory() as tmp:
        path_local = os.path.join(tmp, "local.pdf")
        export_outlier_pdf(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", path=path_local,
                            min_size=20, min_points_to_plot=5, mode="local")
        text = pypdf.PdfReader(path_local).pages[0].extract_text()
        assert "Curve shown" in text
        assert "Flagged against" in text

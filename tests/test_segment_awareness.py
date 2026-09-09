import numpy as np
import pandas as pd

from robustkit import (
    segment_stability_report, segment_benchmark_report,
    hierarchical_segment, apply_by_segment, model_stability_pct,
)
from robustkit.segment_awareness.reports import _build_hierarchy


def make_job_family_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 1000

    job_family = rng.choice(["ENG", "FIN", "ADM"], n, p=[0.6, 0.25, 0.15])
    p_level = rng.choice(["P3", "P4", "P5"], n, p=[0.5, 0.35, 0.15])
    ot = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)

    df = pd.DataFrame({"JobFamily": job_family, "P_niva": p_level, "OT": ot, "age": age, "salary": salary})

    # A few genuinely tiny combinations, forcing fallback behavior
    tiny = pd.DataFrame({
        "JobFamily": ["ADM", "FIN", "ENG"],
        "P_niva": ["P5", "P5", "P5"],
        "OT": ["Yes", "Yes", "Yes"],
        "age": [45, 50, 55],
        "salary": [60000, 65000, 70000],
    })
    return pd.concat([df, tiny], ignore_index=True)


def test_build_hierarchy_drops_from_front():
    hierarchy = _build_hierarchy(["JobFamily", "P_niva", "OT"])
    assert hierarchy == [
        ["JobFamily", "P_niva", "OT"],
        ["P_niva", "OT"],
        ["OT"],
    ]


def test_segment_stability_report_matches_manual_construction():
    """
    The convenience wrapper must produce results identical to manually
    building the hierarchy and calling hierarchical_segment +
    apply_by_segment(model_stability_pct) -- it's meant to save typing,
    not change behavior.
    """
    df = make_job_family_df()

    report = segment_stability_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    hierarchy = [["JobFamily", "P_niva", "OT"], ["P_niva", "OT"], ["OT"]]
    segmented_manual = hierarchical_segment(df, hierarchy, min_size=20)
    report_manual = apply_by_segment(segmented_manual, "segment_id", "age", "salary", model_stability_pct)

    cols = [c for c in report.columns if c != "segment_level"]
    pd.testing.assert_frame_equal(
        report[cols].sort_values("segment").reset_index(drop=True),
        report_manual[cols].sort_values("segment").reset_index(drop=True),
    )


def test_segment_stability_report_includes_segment_level():
    df = make_job_family_df()
    report = segment_stability_report(df, x_col="age", y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], min_size=20)

    assert "segment_level" in report.columns
    # given the injected tiny combinations, at least some fallback should occur
    assert report["segment_level"].max() >= 1


def test_segment_benchmark_report_matches_manual_construction():
    from robustkit import segment_position_report

    df = make_job_family_df()

    report = segment_benchmark_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100,
    )

    hierarchy = [["JobFamily", "P_niva", "OT"], ["P_niva", "OT"], ["OT"]]
    segmented_manual = hierarchical_segment(df, hierarchy, min_size=20)
    report_manual = segment_position_report(segmented_manual, segment_col="segment_id", y_col="salary", x_col="age", n_boot=100)

    cols = [c for c in report.columns if c != "segment_level"]
    pd.testing.assert_frame_equal(
        report[cols].sort_values("segment").reset_index(drop=True),
        report_manual[cols].sort_values("segment").reset_index(drop=True),
    )


def test_segment_benchmark_report_includes_segment_level_and_valid_ci():
    df = make_job_family_df()
    report = segment_benchmark_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], x_col="age", min_size=20, n_boot=100,
    )

    assert "segment_level" in report.columns
    available = report[report["ci_available"]]
    assert (available["ci_lower"] <= available["ci_upper"]).all()


def test_segment_benchmark_report_accepts_custom_benchmark_model():
    """The wrapper must accept the same model-agnostic benchmark_fit
    contract as segment_position_report."""
    df = make_job_family_df()

    class SimpleModel:
        def __init__(self, df):
            X = np.column_stack([np.ones(len(df)), df["age"].to_numpy(dtype=float)])
            self.beta, *_ = np.linalg.lstsq(X, df["salary"].to_numpy(dtype=float), rcond=None)

        def predict(self, df):
            X = np.column_stack([np.ones(len(df)), df["age"].to_numpy(dtype=float)])
            return X @ self.beta

    model = SimpleModel(df)
    report = segment_benchmark_report(
        df, y_col="salary", segment_cols=["JobFamily", "P_niva", "OT"], benchmark_fit=model, min_size=20, n_boot=100,
    )
    assert len(report) > 0
    assert "difference" in report.columns

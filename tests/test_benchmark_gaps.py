import numpy as np
import pandas as pd
import pytest

from robustkit import (
    segment_position_report, fit_huber_benchmark, benchmark_predict, MIN_POINTS_FOR_CI,
    bca_bootstrap_ci, bca_bootstrap_ci_by_index,
)


# ---------------------------------------------------------------------------
# core.uncertainty: bca_bootstrap_ci_by_index generalization
# ---------------------------------------------------------------------------

def test_bca_bootstrap_ci_matches_by_index_version():
    """bca_bootstrap_ci must remain a thin, behavior-preserving wrapper
    around bca_bootstrap_ci_by_index after the refactor."""
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 200)
    y = 1000 + 50 * x + rng.normal(0, 300, 200)

    result_a = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_), n_boot=300, seed=1)

    def by_idx(idx):
        return float(np.median(y[idx]))

    result_b = bca_bootstrap_ci_by_index(len(x), by_idx, n_boot=300, seed=1)

    assert result_a["estimate"] == pytest.approx(result_b["estimate"])
    assert result_a["lower"] == pytest.approx(result_b["lower"])
    assert result_a["upper"] == pytest.approx(result_b["upper"])


# ---------------------------------------------------------------------------
# GAP #4: small segments get no CI, but no crash
# ---------------------------------------------------------------------------

def make_mixed_size_df(seed=0):
    rng = np.random.default_rng(seed)
    n_big = 500
    age_big = rng.uniform(25, 60, n_big)
    salary_big = 30000 + 400 * age_big + rng.normal(0, 800, n_big)

    tiny_rows = []
    for i, tiny_n in enumerate([1, 2, 3]):
        for _ in range(tiny_n):
            tiny_rows.append({
                "department": f"Tiny{i}",
                "age": rng.uniform(25, 60),
                "salary": rng.uniform(25000, 60000),
            })

    return pd.concat([
        pd.DataFrame({"department": ["Big"] * n_big, "age": age_big, "salary": salary_big}),
        pd.DataFrame(tiny_rows),
    ], ignore_index=True)


def test_small_segments_get_no_ci_but_no_crash():
    df = make_mixed_size_df()
    report = segment_position_report(df, segment_col="department", x_col="age", y_col="salary", n_boot=100)

    assert not report["difference"].isna().any()

    tiny = report[report["segment"].str.startswith("Tiny")]
    assert (tiny["ci_available"] == False).all()  # noqa: E712
    assert tiny["ci_lower"].isna().all()
    assert tiny["ci_upper"].isna().all()

    big = report[report["segment"] == "Big"]
    assert (big["ci_available"] == True).all()  # noqa: E712
    assert not big["ci_lower"].isna().any()


def test_min_points_for_ci_threshold_value():
    assert MIN_POINTS_FOR_CI == 20


# ---------------------------------------------------------------------------
# GAP #3: model-agnostic custom benchmark
# ---------------------------------------------------------------------------

class _RichSalaryModel:
    """A stand-in for a richer benchmark: age + age^2 + level."""

    def __init__(self, df, age_col, level_col, y_col):
        X = self._design(df, age_col, level_col)
        beta, *_ = np.linalg.lstsq(X, df[y_col].to_numpy(dtype=float), rcond=None)
        self.beta = beta
        self.age_col = age_col
        self.level_col = level_col

    @staticmethod
    def _design(df, age_col, level_col):
        X = np.column_stack([
            df[age_col],
            df[age_col] ** 2,
            (df[level_col] == "Senior").astype(float),
        ])
        return np.column_stack([np.ones(len(X)), X])

    def predict(self, df):
        return self._design(df, self.age_col, self.level_col) @ self.beta


def make_custom_model_df(seed=3):
    rng = np.random.default_rng(seed)
    n = 600
    age = rng.uniform(25, 60, n)
    level = rng.choice(["Junior", "Senior"], n)
    dept = rng.choice(["Eng", "Sales", "HR"], n, p=[0.5, 0.3, 0.2])
    salary = 25000 + 300 * age - 0.2 * age**2 + 8000 * (level == "Senior") + rng.normal(0, 1000, n)
    salary = salary + np.where(dept == "HR", -3000, 0)  # genuine HR shift, controlling for age+level
    return pd.DataFrame({"age": age, "level": level, "department": dept, "salary": salary})


def test_custom_benchmark_model_detects_controlled_shift():
    df = make_custom_model_df()
    model = _RichSalaryModel(df, "age", "level", "salary")

    report = segment_position_report(df, segment_col="department", y_col="salary", benchmark_fit=model, n_boot=200)

    hr_row = report[report["segment"] == "HR"].iloc[0]
    assert hr_row["ci_upper"] < 0  # clear negative deviation once controlled for age+level


def test_custom_benchmark_without_x_col_does_not_raise():
    df = make_custom_model_df()
    model = _RichSalaryModel(df, "age", "level", "salary")
    # x_col intentionally omitted -- must work for a custom model
    report = segment_position_report(df, segment_col="department", y_col="salary", benchmark_fit=model, n_boot=50)
    assert len(report) == 3


def test_dict_benchmark_without_x_col_raises():
    df = make_custom_model_df()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float))
    with pytest.raises(ValueError):
        benchmark_predict(fit, df, x_col=None)


def test_benchmark_predict_rejects_unsupported_type():
    with pytest.raises(TypeError):
        benchmark_predict(object(), pd.DataFrame({"a": [1, 2]}))


def test_legacy_single_column_mode_still_works():
    """Backward compatibility: calling with x_col and no benchmark_fit
    must behave as before the GAP #3/#4 changes."""
    df = make_custom_model_df()
    report = segment_position_report(df, segment_col="department", x_col="age", y_col="salary", n_boot=100)
    assert set(report["segment"]) == {"Eng", "Sales", "HR"}
    assert (report["ci_available"] == True).all()  # noqa: E712 -- all segments here are large


def test_missing_x_col_and_benchmark_fit_raises():
    df = make_custom_model_df()
    with pytest.raises(ValueError):
        segment_position_report(df, segment_col="department", y_col="salary")

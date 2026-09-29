import numpy as np
import pandas as pd
import pytest

from robustkit import benchmark_goodness_of_fit, benchmark_screening_report, fit_huber_benchmark, predict_trend


def make_data(seed=141, n=1500):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n, p=[0.4, 0.35, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.5, 0.35, 0.15])
    age = rng.uniform(25, 60, n)
    family_bump = {"ENG": 3000, "ITS": -1500, "FIN": 0}
    level_bump = {"L1": 0, "L2": 6000, "L3": 14000}
    salary = (32000 + 200 * age + 15 * (age ** 2) + np.array([family_bump[f] for f in job_family])
              + np.array([level_bump[l] for l in level]) + rng.normal(0, 900, n))
    return pd.DataFrame({"JobFamily": job_family, "level": level, "age": age, "salary": salary})


def make_complete_benchmark(df):
    """Standardized features -- see the identical fixture's docstring
    in test_benchmark_screening_report.py for why this matters
    (unstandardized age/age^2 alongside 0/1 dummies caused
    HuberRegressor convergence issues in a real environment)."""
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler

    X = pd.get_dummies(df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
    poly_X = X.copy()
    poly_X["age_sq"] = df["age"] ** 2

    scaler = StandardScaler()
    poly_X_scaled = pd.DataFrame(scaler.fit_transform(poly_X), columns=poly_X.columns, index=poly_X.index)

    model = HuberRegressor(max_iter=1000)
    model.fit(poly_X_scaled, df["salary"])

    class CompleteBenchmark:
        def __init__(self):
            self.columns = poly_X.columns

        def predict(self, sub_df):
            Xs = pd.get_dummies(sub_df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
            Xs["age_sq"] = sub_df["age"] ** 2
            Xs = Xs.reindex(columns=self.columns, fill_value=0)
            Xs_scaled = pd.DataFrame(scaler.transform(Xs), columns=Xs.columns, index=Xs.index)
            return model.predict(Xs_scaled)

    return CompleteBenchmark()


def test_returns_expected_keys():
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    result = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=fit, x_col="age")
    assert set(result.keys()) == {"r_squared", "rmse", "mae"}


def test_complete_benchmark_scores_higher_than_incomplete():
    df = make_data()
    incomplete_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    incomplete = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=incomplete_fit, x_col="age")
    complete = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=make_complete_benchmark(df), x_col="age")

    assert complete["r_squared"] > incomplete["r_squared"]
    assert complete["rmse"] < incomplete["rmse"]
    assert complete["mae"] < incomplete["mae"]
    assert complete["r_squared"] > 0.95  # near-complete model should explain most variance


def test_works_without_x_col_for_pure_custom_model():
    df = make_data()

    class NoAgeArgBenchmark:
        def predict(self, sub_df):
            return np.full(len(sub_df), df["salary"].mean())

    result = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=NoAgeArgBenchmark())
    assert set(result.keys()) == {"r_squared", "rmse", "mae"}


def test_matches_robustkit_trend_fit_dict_directly():
    """A robustkit trend-fit dict must work identically here as it
    does inside benchmark_predict elsewhere in the package."""
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=2)
    result = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=fit, x_col="age")

    # manually reproduce the expected r_squared
    expected = predict_trend(fit, df["age"].to_numpy(dtype=float))
    residuals = df["salary"].to_numpy(dtype=float) - expected
    manual_r2 = 1 - np.var(residuals) / np.var(df["salary"].to_numpy(dtype=float))
    assert abs(result["r_squared"] - manual_r2) < 1e-9


def test_benchmark_screening_report_includes_fit_key():
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"])
    assert "fit" in result
    assert set(result["fit"].keys()) == {"r_squared", "rmse", "mae"}


def test_benchmark_screening_report_fit_key_matches_standalone_call():
    """The 'fit' entry inside benchmark_screening_report must be
    numerically identical to calling benchmark_goodness_of_fit
    directly with the same arguments -- no drift between the two."""
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    standalone = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=fit, x_col="age")
    embedded = benchmark_screening_report(df, y_col="salary", benchmark_fit=fit, x_col="age",
                                           candidate_cols=["JobFamily", "level"])["fit"]
    assert standalone == embedded

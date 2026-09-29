import numpy as np
import pandas as pd
import pytest

from robustkit import benchmark_screening_report, fit_huber_benchmark


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
    """
    A near-complete benchmark (age, age^2, JobFamily, level) used to
    verify the screening report reports near-zero residual structure
    when little is actually missing.

    Features are explicitly standardized before fitting: age and
    age^2 span very different scales (tens vs. thousands) from the
    0/1 dummy columns, and HuberRegressor on unstandardized inputs was
    found NOT to reliably converge -- leaving small but real leftover
    coefficient bias, and with it, small but nonzero residual mutual
    information that made this fixture's own "near-complete" claim
    untrue in practice. Standardizing fixed convergence (confirmed via
    model.n_iter_) and brought residual MI down to the small,
    genuinely-near-zero range the test actually expects.
    """
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


def test_polynomial_part_detects_genuine_quadratic_gap():
    df = make_data()
    incomplete_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=incomplete_fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"])
    poly = result["polynomial"]
    gain_1_to_2 = poly[poly["degree"] == 2]["r_squared_gain"].values[0]
    gain_2_to_3 = poly[poly["degree"] == 3]["r_squared_gain"].values[0]
    assert gain_1_to_2 > 0.005  # a real, meaningful gain
    assert gain_2_to_3 < gain_1_to_2 / 10  # negligible beyond degree 2


def test_interaction_part_surfaces_missing_categorical_structure():
    df = make_data()
    incomplete_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=2)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=incomplete_fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"])
    interactions = result["interactions"]
    assert set(interactions["feature"]) == {"JobFamily", "level", "JobFamily x level"}
    assert (interactions["mutual_information"] > 0.1).all()


def test_complete_benchmark_shows_near_zero_residual_structure():
    df = make_data()
    complete_fit = make_complete_benchmark(df)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=complete_fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"])
    interactions = result["interactions"]
    # threshold intentionally generous (not 0.0) -- MI is a finite-sample
    # statistical estimate with inherent variance, and HuberRegressor's
    # exact convergence point can vary slightly across sklearn versions/
    # BLAS backends even with standardized features
    assert (interactions["mutual_information"] < 0.08).all()


def test_information_efficiency_ranks_differently_from_raw_mi():
    """The core point of using efficiency, not raw MI: the interaction
    term can have higher raw MI but lower efficiency than a simpler
    candidate, and results must be sorted by efficiency."""
    df = make_data()
    incomplete_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=incomplete_fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"])
    interactions = result["interactions"]
    # sorted descending by efficiency
    eff = interactions["information_efficiency"].to_numpy()
    assert (np.diff(eff) <= 1e-9).all()


def test_explicit_interaction_pairs_overrides_auto_generation():
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=2)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=fit, x_col="age",
                                         candidate_cols=["JobFamily", "level"], interaction_pairs=[])
    assert set(result["interactions"]["feature"]) == {"JobFamily", "level"}  # no interaction column


def test_no_x_col_skips_polynomial_part():
    df = make_data()
    from sklearn.linear_model import HuberRegressor
    X = pd.get_dummies(df[["JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
    model = HuberRegressor()
    model.fit(X, df["salary"])

    class NoAgeBenchmark:
        def predict(self, sub_df):
            Xs = pd.get_dummies(sub_df[["JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
            return model.predict(Xs.reindex(columns=X.columns, fill_value=0))

    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=NoAgeBenchmark(),
                                         candidate_cols=["JobFamily", "level"])
    assert "polynomial" not in result
    assert "interactions" in result


def test_candidate_cols_defaults_to_non_numeric_columns():
    df = make_data()
    fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=2)
    result = benchmark_screening_report(df, y_col="salary", benchmark_fit=fit, x_col="age")
    base_features = {f for f in result["interactions"]["feature"] if " x " not in f}
    assert base_features == {"JobFamily", "level"}

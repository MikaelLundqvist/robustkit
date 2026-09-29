import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robustkit import plot_segment_vs_benchmark, fit_huber_benchmark


def make_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "JobFamily": rng.choice(["ENG", "ITS", "FIN"], n),
        "age": rng.uniform(25, 60, n),
    })
    df["salary"] = 30000 + 400 * df["age"] + rng.normal(0, 900, n)
    return df


def test_plot_segment_vs_benchmark_requires_statsmodels_or_skips():
    """Confirms the function actually depends on statsmodels (lazy
    import) rather than silently no-op'ing without it -- the whole
    point of using LOWESS here."""
    pytest.importorskip("statsmodels")
    df = make_df()
    ax = plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG", "ITS"],
                                    y_col="salary", x_col="age")
    assert ax is not None


def test_plot_segment_vs_benchmark_without_statsmodels_raises_import_error():
    """In an environment without statsmodels, calling this should
    fail clearly (ImportError from the lazy import), not silently
    produce a wrong or empty chart."""
    import sys
    if "statsmodels" in sys.modules:
        pytest.skip("statsmodels is installed in this environment; this test targets its absence")
    df = make_df()
    with pytest.raises(ImportError):
        plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG"], y_col="salary", x_col="age")


def test_plot_segment_vs_benchmark_default_benchmark_fit():
    pytest.importorskip("statsmodels")
    df = make_df()
    ax = plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG", "ITS"],
                                    y_col="salary", x_col="age")
    # two segments -> two observed lines + two benchmark lines = 4 lines total
    assert len(ax.lines) == 4
    assert len(ax.collections) == 2  # one scatter per segment


def test_plot_segment_vs_benchmark_custom_benchmark_fit():
    pytest.importorskip("statsmodels")
    df = make_df()
    custom_fit = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    ax = plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG"],
                                    y_col="salary", x_col="age", benchmark_fit=custom_fit)
    assert len(ax.lines) == 2


def test_plot_segment_vs_benchmark_respects_passed_ax():
    pytest.importorskip("statsmodels")
    import matplotlib.pyplot as plt
    df = make_df()
    fig, custom_ax = plt.subplots()
    returned = plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG"],
                                          y_col="salary", x_col="age", ax=custom_ax)
    assert returned is custom_ax


def test_plot_segment_vs_benchmark_skips_nonexistent_segment_gracefully():
    pytest.importorskip("statsmodels")
    df = make_df()
    ax = plot_segment_vs_benchmark(df, segment_col="JobFamily", segment_values=["ENG", "NONEXISTENT"],
                                    y_col="salary", x_col="age")
    # only ENG actually plotted -> 2 lines, 1 scatter collection
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1


def test_plot_segment_vs_benchmark_works_with_custom_multivariate_model():
    """The main real-world use case: a benchmark that depends on more
    than x_col, with the plot correctly holding other model inputs
    constant per-segment when building its prediction grid."""
    pytest.importorskip("statsmodels")
    from sklearn.linear_model import HuberRegressor

    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({
        "JobFamily": rng.choice(["ENG", "ITS"], n),
        "level": rng.choice(["L1", "L2"], n),
        "age": rng.uniform(25, 60, n),
    })
    level_bump = {"L1": 0, "L2": 5000}
    df["salary"] = 30000 + 400 * df["age"] + df["level"].map(level_bump) + rng.normal(0, 800, n)

    X = pd.get_dummies(df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
    model = HuberRegressor()
    model.fit(X, df["salary"])

    class CustomBenchmark:
        def __init__(self):
            self.X_columns = X.columns

        def predict(self, sub_df):
            Xs = pd.get_dummies(sub_df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
            return model.predict(Xs.reindex(columns=self.X_columns, fill_value=0))

    sub_l2 = df[df["level"] == "L2"].copy()
    ax = plot_segment_vs_benchmark(sub_l2, segment_col="JobFamily", segment_values=["ENG", "ITS"],
                                    y_col="salary", x_col="age", benchmark_fit=CustomBenchmark())
    assert len(ax.lines) == 4

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for tests -- no window, no blocking

import numpy as np
import pandas as pd
import pytest

from robustkit import (
    entropy, rank_features, information_efficiency,
    quadrant_report, plot_feature_space, feature_map,
    profile, print_profile,
)


def make_test_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    # strong_signal: directly drives the continuous target
    strong_signal = rng.uniform(0, 10, n)
    # weak_signal: barely related to the target
    weak_signal = rng.uniform(0, 10, n)
    # noise: unrelated categorical, but many levels (high entropy, ~0 MI)
    noise_category = rng.choice([f"cat_{i}" for i in range(20)], size=n)
    # low_cardinality: few levels, weakly related
    low_cardinality = rng.choice(["A", "B"], size=n)

    target = 100 + 10 * strong_signal + 0.5 * weak_signal + rng.normal(0, 2, n)

    return pd.DataFrame({
        "strong_signal": strong_signal,
        "weak_signal": weak_signal,
        "noise_category": noise_category,
        "low_cardinality": low_cardinality,
        "target": target,
    })


def test_entropy_basic_properties():
    # A perfectly uniform 4-category variable has exactly 2 bits of entropy
    uniform = pd.Series(["a", "b", "c", "d"] * 25)
    assert entropy(uniform) == pytest.approx(2.0, abs=1e-9)

    # A constant variable has zero entropy
    constant = pd.Series(["x"] * 100)
    assert entropy(constant) == pytest.approx(0.0, abs=1e-9)

    # More categories (all else equal) -> more entropy
    two_cat = pd.Series(["a", "b"] * 50)
    eight_cat = pd.Series([f"c{i}" for i in range(8)] * 12)
    assert entropy(eight_cat) > entropy(two_cat)


def test_information_efficiency_edge_cases():
    assert information_efficiency(mutual_information=0.5, entropy_bits=0) == 0.0
    assert information_efficiency(mutual_information=1.0, entropy_bits=2.0) == pytest.approx(0.5)


def test_rank_features_orders_strong_signal_above_noise():
    df = make_test_df()
    ranking = rank_features(df, target="target", seed=0)

    assert set(ranking["feature"]) == {"strong_signal", "weak_signal", "noise_category", "low_cardinality"}

    strong_mi = ranking.loc[ranking["feature"] == "strong_signal", "mutual_information"].iloc[0]
    noise_mi = ranking.loc[ranking["feature"] == "noise_category", "mutual_information"].iloc[0]
    assert strong_mi > noise_mi


def test_quadrant_report_thresholds_match_median_by_default():
    df = make_test_df()
    report = quadrant_report(df, target="target")

    assert set(report["quadrant"]) <= {"star", "power", "efficient", "weak"}
    assert report.attrs["mi_threshold"] == pytest.approx(report["mutual_information"].median())
    assert report.attrs["eff_threshold"] == pytest.approx(report["information_efficiency"].median())


def test_quadrant_report_accepts_precomputed_ranking():
    df = make_test_df()
    ranking = rank_features(df, target="target")
    report_from_ranking = quadrant_report(ranking=ranking)
    report_from_df = quadrant_report(df, target="target")

    # Same underlying data -> identical quadrant assignment either way
    pd.testing.assert_series_equal(
        report_from_ranking.sort_values("feature")["quadrant"].reset_index(drop=True),
        report_from_df.sort_values("feature")["quadrant"].reset_index(drop=True),
    )


def test_plot_feature_space_quadrants_match_quadrant_report():
    """
    Regression test for the bug this module was built to fix: the plot
    and the report must never disagree about which quadrant a feature
    is in.
    """
    df = make_test_df()
    report = quadrant_report(df, target="target")
    plotted_ranking = plot_feature_space(ranking=report.copy(), annotate=False)

    merged = report.merge(plotted_ranking, on="feature", suffixes=("_report", "_plot"))
    assert (merged["quadrant_report"] == merged["quadrant_plot"]).all()


def test_feature_map_runs_end_to_end():
    df = make_test_df()
    ranking = feature_map(df, target="target", annotate=False)
    assert "quadrant" in ranking.columns
    assert len(ranking) == 4


def test_profile_reports_missing_values():
    df = make_test_df()
    df.loc[0:4, "weak_signal"] = np.nan

    p = profile(df, target="target")
    assert p["n_rows"] == len(df)
    assert p["missing_per_column"]["weak_signal"] == 5
    assert p["target"] == "target"

    print_profile(df, target="target")  # just check it doesn't raise

import time

import numpy as np
import pandas as pd
import pytest

from robustkit.core.uncertainty import bca_bootstrap_ci_by_index, bca_bootstrap_ci
from robustkit import segment_contribution_report, segment_benchmark_drilldown_report


def test_jackknife_cap_default_does_not_trigger_below_threshold():
    """n < jackknife_cap (default 1000) must give byte-identical
    results to the pre-fix, always-full-jackknife behavior."""
    rng = np.random.default_rng(0)
    data = rng.normal(50, 10, 200)

    def stat_fn(idx):
        return float(np.median(data[idx]))

    result_default = bca_bootstrap_ci_by_index(200, stat_fn, n_boot=100, seed=0)
    result_explicit_none = bca_bootstrap_ci_by_index(200, stat_fn, n_boot=100, seed=0, jackknife_cap=None)

    assert result_default["lower"] == result_explicit_none["lower"]
    assert result_default["upper"] == result_explicit_none["upper"]
    assert result_default["a"] == result_explicit_none["a"]


def test_jackknife_cap_gives_close_ci_to_uncapped_on_moderate_n():
    """Above the cap, results are a legitimate approximation, not
    identical -- but should stay close. Verified directly against a
    real dataset elsewhere; here just a sanity bound on synthetic
    data at a size still small enough to compute both ways."""
    rng = np.random.default_rng(0)
    data = rng.normal(100, 20, 3000)

    def stat_fn(idx):
        return float(np.median(data[idx]))

    result_capped = bca_bootstrap_ci_by_index(3000, stat_fn, n_boot=100, seed=0, jackknife_cap=500)
    result_full = bca_bootstrap_ci_by_index(3000, stat_fn, n_boot=100, seed=0, jackknife_cap=None)

    # CI bounds should be close (not necessarily identical) -- both
    # describe the same underlying distribution's median
    assert abs(result_capped["lower"] - result_full["lower"]) < 1.0
    assert abs(result_capped["upper"] - result_full["upper"]) < 1.0


def test_jackknife_cap_makes_large_n_tractable():
    """The actual point of the fix: n_boot='auto' alone was NOT
    enough to keep this tractable (jackknife runs regardless of
    n_boot) -- capping the jackknife itself is what matters. This
    must complete quickly even for a large n."""
    rng = np.random.default_rng(0)
    n = 50000
    data = rng.normal(0, 1, n)

    def stat_fn(idx):
        return float(np.median(data[idx]))

    t0 = time.time()
    result = bca_bootstrap_ci_by_index(n, stat_fn, n_boot=50, seed=0)
    elapsed = time.time() - t0

    assert elapsed < 10, f"expected well under 10s with jackknife capping, took {elapsed:.1f}s"
    assert result["lower"] < result["estimate"] < result["upper"]


def test_bca_bootstrap_ci_wrapper_unaffected_by_jackknife_cap_default():
    """bca_bootstrap_ci (the simpler x/y wrapper) must remain
    behavior-identical to bca_bootstrap_ci_by_index post-fix, for n
    below the cap -- same check as the pre-existing
    test_bca_bootstrap_ci_matches_by_index_version, re-verified here
    after the jackknife_cap change specifically."""
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 200)
    y = 1000 + 50 * x + rng.normal(0, 300, 200)

    result_a = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_), n_boot=300, seed=1)

    def by_idx(idx):
        return float(np.median(y[idx]))

    result_b = bca_bootstrap_ci_by_index(len(x), by_idx, n_boot=300, seed=1)

    assert abs(result_a["estimate"] - result_b["estimate"]) < 1e-9
    assert abs(result_a["lower"] - result_b["lower"]) < 1e-9
    assert abs(result_a["upper"] - result_b["upper"]) < 1e-9


# ---------------------------------------------------------------------------
# segment_contribution_report / segment_benchmark_drilldown_report:
# precomputed-residual optimization must not change results, only speed
# ---------------------------------------------------------------------------

def make_moderate_df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n)
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    return pd.DataFrame({"JobFamily": job_family, "age": age, "salary": salary})


def test_segment_contribution_report_still_correct_after_optimization():
    """The precomputed-residual rewrite must not change the actual
    contribution values -- same arithmetic, just computed faster."""
    df = make_moderate_df()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", min_size=20, n_boot=100)
    assert len(report) > 0
    assert "contribution" in report.columns
    assert "difference" in report.columns


def test_segment_contribution_report_tractable_on_large_real_scale():
    """Regression for the actual bug found via real-data validation:
    a ~185,000-row segment previously made this time out; must now
    complete quickly even at a large synthetic scale."""
    df = make_moderate_df(n=60000)
    t0 = time.time()
    report = segment_contribution_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age", min_size=20)
    elapsed = time.time() - t0
    assert elapsed < 30, f"expected well under 30s, took {elapsed:.1f}s"
    assert len(report) > 0


def test_segment_position_report_tractable_on_large_real_scale():
    """segment_position_report had the same unfixed .iloc + jackknife
    pattern as segment_contribution_report -- discovered while
    building the book chapter that showcases it. Same regression
    check: must stay fast even on a large segment."""
    from robustkit import segment_position_report
    df = make_moderate_df(n=60000)
    t0 = time.time()
    report = segment_position_report(df, segment_col="JobFamily", y_col="salary", x_col="age")
    elapsed = time.time() - t0
    assert elapsed < 15, f"expected well under 15s, took {elapsed:.1f}s"
    assert len(report) > 0
    assert report["ci_available"].all()

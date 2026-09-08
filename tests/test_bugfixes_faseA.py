import time

import numpy as np
import pandas as pd
import pytest

from robustkit import rank_features, quadrant_report, rank_communicative_pairs, bootstrap_band
from robustkit.core.uncertainty import _resolve_n_boot


# ---------------------------------------------------------------------------
# Bug 1: information module crashed on real pandas Categorical columns
# with missing values (observed on OpenML's Boston Housing "CHAS" column)
# ---------------------------------------------------------------------------

def make_categorical_with_na_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    chas = pd.Series(pd.Categorical(rng.choice(["0", "1"], n)))
    chas.iloc[:20] = np.nan  # NaN within an existing Categorical, no "Missing" category defined

    return pd.DataFrame({
        "CHAS": chas,
        "RM": rng.uniform(4, 8, n),
        "LSTAT": rng.uniform(2, 35, n),
        "MEDV": rng.uniform(10, 50, n),
    })


def test_rank_features_handles_categorical_with_missing_values():
    df = make_categorical_with_na_df()
    assert df["CHAS"].dtype.name == "category"
    assert df["CHAS"].isna().sum() == 20

    # This used to raise: TypeError: Cannot setitem on a Categorical
    # with a new category ('Missing'), set the categories first
    ranking = rank_features(df, target="MEDV")
    assert set(ranking["feature"]) == {"CHAS", "RM", "LSTAT"}


def test_quadrant_report_handles_categorical_with_missing_values():
    df = make_categorical_with_na_df()
    report = quadrant_report(df, target="MEDV")
    assert "quadrant" in report.columns


def test_rank_communicative_pairs_handles_categorical_with_missing_values():
    df = make_categorical_with_na_df()
    pairs = rank_communicative_pairs(df, target="MEDV", features=["CHAS", "RM", "LSTAT"])
    assert len(pairs) == 3  # C(3,2)


# ---------------------------------------------------------------------------
# Bug 2: bootstrap_band became very slow on large datasets (n~54000,
# n_boot=200 -- each iteration refits a full Huber model)
# ---------------------------------------------------------------------------

def test_resolve_n_boot_schedule():
    assert _resolve_n_boot(100, "auto") == 500     # small n: unchanged from the old fixed default
    assert _resolve_n_boot(6000, "auto") == 200
    assert _resolve_n_boot(15000, "auto") == 100
    assert _resolve_n_boot(54000, "auto") == 50
    assert _resolve_n_boot(54000, 500) == 500       # explicit override always respected, regardless of n


def test_bootstrap_band_auto_default_produces_valid_band():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 300)
    y = 1000 + 50 * x + rng.normal(0, 300, 300)

    band = bootstrap_band(x, y)  # n_boot defaults to "auto" now
    assert np.all(band["lower"] <= band["median"])
    assert np.all(band["median"] <= band["upper"])


def test_bootstrap_band_large_n_completes_quickly():
    """
    Regression test for the reported slowdown: n_boot='auto' should
    keep large-dataset runs tractable. Not a strict timing assertion
    (machine-dependent), but a generous upper bound to catch a
    regression back to the old fixed-n_boot=500 behavior.
    """
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.uniform(0.2, 3.0, n)
    y = 3000 * x**2 + rng.normal(0, 1000, n)

    start = time.time()
    band = bootstrap_band(x, y)  # "auto" -> n_boot=100 at this n
    elapsed = time.time() - start

    assert np.all(band["lower"] <= band["upper"])
    assert elapsed < 30, f"bootstrap_band took {elapsed:.1f}s on n={n} -- expected well under 30s with auto n_boot"


def test_bootstrap_band_explicit_n_boot_opts_out_of_auto():
    rng = np.random.default_rng(2)
    x = rng.uniform(20, 60, 100)
    y = 1000 + 50 * x + rng.normal(0, 300, 100)

    # Explicit n_boot must be respected exactly, not overridden by the schedule
    band = bootstrap_band(x, y, n_boot=17, n_points=10)
    assert band["lower"].shape == (10,)

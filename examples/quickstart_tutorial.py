"""
robustkit quickstart tutorial
==============================

A self-contained walkthrough of every function in robustkit.core,
run against a small synthetic dataset generated inline (no external
data files needed). Run with:

    python examples/quickstart_tutorial.py
"""

import numpy as np

from robustkit import (
    fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend,
    model_stability_pct, cooks_diagnostic, cook_impact,
    bootstrap_band, bca_bootstrap_ci,
)


def make_example_data(n=200, seed=0, n_outliers=4):
    """
    A small synthetic (x, y) pair with a concave trend and a handful
    of injected outliers, so every diagnostic below has something to
    find.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 60, n)
    y = 1000 + 60 * x - 0.5 * x**2 + rng.normal(0, 400, n)

    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    y[outlier_idx] += rng.choice([-1, 1], size=n_outliers) * rng.uniform(4000, 7000, n_outliers)

    return x, y


def main():
    x, y = make_example_data()
    print(f"Example dataset: {len(x)} points, {4} injected outliers\n")

    # ------------------------------------------------------------------
    # 1. Fit and compare three trend curves
    # ------------------------------------------------------------------
    print("=== 1. Trend fitting ===")
    huber_fit = fit_huber_trend(x, y, degree=2)
    tukey_fit = fit_tukey_trend(x, y, degree=2)
    ols_fit = fit_ols_trend(x, y, degree=2)

    check_points = [25, 40, 55]
    print("Predictions at x =", check_points)
    print("  Huber:", predict_trend(huber_fit, check_points).round(0))
    print("  Tukey:", predict_trend(tukey_fit, check_points).round(0))
    print("  OLS:  ", predict_trend(ols_fit, check_points).round(0))
    print()

    # ------------------------------------------------------------------
    # 2. Does the conclusion survive a change of method?
    # ------------------------------------------------------------------
    print("=== 2. Model stability ===")
    stability = model_stability_pct(x, y)
    print(f"Median % spread between methods: {stability['median_pct_diff']:.1f}%")
    print(f"95th percentile % spread:        {stability['p95_pct_diff']:.1f}%")
    print(f"Max % spread:                    {stability['max_pct_diff']:.1f}%")
    print("(Large spread here is expected -- this example has outliers by design.)\n")

    # ------------------------------------------------------------------
    # 3. Which points are influential, and does it matter?
    # ------------------------------------------------------------------
    print("=== 3. Influence diagnostics ===")
    diag = cooks_diagnostic(x, y)
    print(f"Flagged {len(diag['flagged_indices'])} of {len(x)} points "
          f"(Cook's distance > {diag['threshold']:.4f})")

    if len(diag["flagged_indices"]) > 0:
        impact = cook_impact(x, y, diag["flagged_indices"])
        print(f"Median % change in curve if flagged points removed: "
              f"{impact['median_pct_change']:.1f}%")
        print(f"Max % change:                                       "
              f"{impact['max_pct_change']:.1f}%")
    print()

    # ------------------------------------------------------------------
    # 4. Uncertainty
    # ------------------------------------------------------------------
    print("=== 4. Uncertainty ===")
    band = bootstrap_band(x, y, n_boot=200)
    mid = len(band["grid"]) // 2
    print(f"Percentile bootstrap band at x={band['grid'][mid]:.0f}: "
          f"[{band['lower'][mid]:.0f}, {band['upper'][mid]:.0f}]")

    ci = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_), n_boot=500)
    print(f"BCa CI for median(y): [{ci['lower']:.0f}, {ci['upper']:.0f}] "
          f"(point estimate: {ci['estimate']:.0f})")
    print(f"  bias correction z0={ci['z0']:.3f}, acceleration a={ci['a']:.3f}")


if __name__ == "__main__":
    main()

"""
Validation runner for robustkit.core against a real dataset's
continuous x/y pair.
"""

import numpy as np
import pandas as pd

import robustkit as rk

from .common import section, safe_run


def run_core_module(df, x_col, y_col):
    section(f"robustkit.core -- x={x_col}, y={y_col}")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
    print(f"  n = {len(x)} (after dropping missing values)")

    fit = safe_run("fit_huber_trend", lambda: rk.fit_huber_trend(x, y, degree=2))
    safe_run("fit_tukey_trend", lambda: rk.fit_tukey_trend(x, y, degree=2))
    safe_run("fit_ols_trend", lambda: rk.fit_ols_trend(x, y, degree=2))

    stability = safe_run("model_stability_pct", lambda: rk.model_stability_pct(x, y))
    if stability:
        print(f"         median_pct_diff={stability['median_pct_diff']:.2f}%, "
              f"p95={stability['p95_pct_diff']:.2f}%, max={stability['max_pct_diff']:.2f}%")

    diag = safe_run("cooks_diagnostic", lambda: rk.cooks_diagnostic(x, y))
    if diag is not None:
        n_flagged = len(diag["flagged_indices"])
        print(f"         {n_flagged} of {len(x)} points flagged (threshold={diag['threshold']:.4f})")
        if n_flagged > 0:
            impact = safe_run("cook_impact", lambda: rk.cook_impact(x, y, diag["flagged_indices"]))
            if impact:
                print(f"         median_pct_change={impact['median_pct_change']:.2f}%, "
                      f"max_pct_change={impact['max_pct_change']:.2f}%")

    safe_run("bootstrap_band", lambda: rk.bootstrap_band(x, y, n_boot=200))
    safe_run("bca_bootstrap_ci", lambda: rk.bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_), n_boot=300))

    gof = safe_run("goodness_of_fit (degree=2)", lambda: rk.goodness_of_fit(x, y, degree=2))
    if gof:
        print(f"         R^2={gof['r_squared']:.3f}, RMSE={gof['rmse']:.1f}, MAE={gof['mae']:.1f}")

    comparison = safe_run("compare_polynomial_degrees", lambda: rk.compare_polynomial_degrees(x, y, degrees=(1, 2, 3, 4)))
    if comparison is not None:
        print(comparison.to_string(index=False))

    if fit is not None:
        check_x = np.percentile(x, [10, 50, 90])
        rates = safe_run("trend_derivative", lambda: rk.trend_derivative(fit, check_x))
        if rates is not None:
            print(f"         growth rate at x p10/p50/p90 ({check_x.round(1)}): {rates.round(3)}")

    return x, y

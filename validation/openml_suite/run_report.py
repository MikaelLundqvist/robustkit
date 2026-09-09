"""
Validation runner for robustkit.report.
"""

import robustkit as rk

from .common import section, safe_run


def run_report_module(x, y):
    section("robustkit.report -- analyst vs. publisher view")
    safe_run("plot_analyst_view", lambda: rk.plot_analyst_view(x, y, n_boot=200, show_points=False))
    binned = safe_run("plot_publisher_view", lambda: rk.plot_publisher_view(x, y, n_bins=8, show_points=False))
    if binned is not None:
        print(binned[["x_center", "n", "q1", "median", "q3", "dispersion_ratio"]].to_string(index=False))

    section("robustkit.report -- combined Huber+IQR view (plot_huber_iqr)")

    result_bin = safe_run(
        "plot_huber_iqr (grouping='bin', default)",
        lambda: rk.plot_huber_iqr(x, y, degree=2, bins=10, grouping="bin"),
    )
    if result_bin is not None:
        n_nan = int(result_bin["binned"][["q1", "q3"]].isna().any(axis=1).sum())
        print(f"  bin mode: {len(result_bin['binned'])} bins, {n_nan} with missing IQR (should be 0)")

    result_unique = safe_run(
        "plot_huber_iqr (grouping='unique', workbook-compatible mode)",
        lambda: rk.plot_huber_iqr(x, y, degree=2, grouping="unique", min_n_for_iqr=5),
    )
    if result_unique is not None:
        binned_u = result_unique["binned"]
        n_with_iqr = int(binned_u[["q1", "q3"]].notna().all(axis=1).sum())
        n_without_iqr = len(binned_u) - n_with_iqr
        print(f"  unique mode: {len(binned_u)} distinct x values, "
              f"{n_with_iqr} with IQR shown (n>=5), {n_without_iqr} without (n<5)")

    custom_title = safe_run(
        "plot_huber_iqr (custom title)",
        lambda: rk.plot_huber_iqr(x, y, degree=2, bins=10, title=f"Validation run ({len(x)} observations)"),
    )

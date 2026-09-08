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

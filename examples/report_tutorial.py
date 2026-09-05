"""
robustkit.report quickstart tutorial
=======================================

A self-contained walkthrough of the analyst view vs. publisher view
distinction, and dispersion measures, run against a small synthetic
dataset with deliberately growing spread. Run with:

    python examples/report_tutorial.py
"""

import numpy as np

from robustkit import (
    iqr, dispersion_ratio, dispersion_by_bin,
    plot_analyst_view, plot_publisher_view,
)


def make_example_data(n=1000, seed=0):
    """
    Right-skewed, salary-like data where spread genuinely grows with
    age -- not just the trend itself, but how spread out individual
    values are around it.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(22, 65, n)
    base = 25000 + 500 * age
    noise_sigma = 0.05 + 0.003 * age  # spread grows with age
    salary = base * rng.lognormal(mean=0, sigma=noise_sigma)
    return age, salary


def main():
    age, salary = make_example_data()
    print(f"Example dataset: {len(age)} points\n")

    # ------------------------------------------------------------------
    # 1. Dispersion measures
    # ------------------------------------------------------------------
    print("=== 1. Dispersion measures ===")
    print(f"Overall IQR: {iqr(salary):.0f}")
    print(f"Overall dispersion_ratio (IQR/median): {dispersion_ratio(salary):.3f}")
    print()

    binned = dispersion_by_bin(age, salary, n_bins=8)
    print(binned[["x_center", "n", "q1", "median", "q3", "dispersion_ratio"]].to_string(index=False))
    print()
    print("Note: dispersion_ratio rises across age bins in this example --")
    print("the population genuinely gets more spread out with age, not just")
    print("higher on average.\n")

    # ------------------------------------------------------------------
    # 2. Analyst view: confidence in the ESTIMATE (shrinks with more data)
    # ------------------------------------------------------------------
    print("=== 2. Analyst view vs. publisher view ===")
    band_small = plot_analyst_view(age, salary, n_boot=200, show_points=False)
    mid = len(band_small["grid"]) // 2
    width_small = band_small["upper"][mid] - band_small["lower"][mid]
    print(f"Analyst-view CI width at a sample point (n={len(age)}): {width_small:.0f}")

    # Same underlying distribution, 20x more data
    rng = np.random.default_rng(1)
    age_big = np.concatenate([age] * 20)
    salary_big = np.concatenate([salary] * 20) * (1 + rng.normal(0, 0.001, len(age) * 20))

    band_big = plot_analyst_view(age_big, salary_big, n_boot=100, show_points=False)
    width_big = band_big["upper"][mid] - band_big["lower"][mid]
    print(f"Analyst-view CI width with 20x data (n={len(age_big)}): {width_big:.0f}")
    print("-> The confidence band shrinks: more data means more certainty")
    print("   about where the true trend lies.\n")

    # ------------------------------------------------------------------
    # 3. Publisher view: actual SPREAD in the population (does not shrink)
    # ------------------------------------------------------------------
    pub_small = plot_publisher_view(age, salary, n_bins=8, show_points=False)
    pub_big = plot_publisher_view(age_big, salary_big, n_bins=8, show_points=False)

    iqr_small = (pub_small["q3"] - pub_small["q1"]).iloc[3]
    iqr_big = (pub_big["q3"] - pub_big["q1"]).iloc[3]
    print(f"Publisher-view IQR at a bin with n={len(age)}: {iqr_small:.0f}")
    print(f"Publisher-view IQR with 20x data: {iqr_big:.0f}")
    print("-> The IQR band stays essentially unchanged: it reflects real")
    print("   dispersion in the population, which more data doesn't reduce.\n")

    print("=== Summary ===")
    print("Same dataset, two different questions:")
    print("  plot_analyst_view()   -> 'How sure are we about the trend?'")
    print("  plot_publisher_view() -> 'How spread out are people's actual values?'")
    print()
    print("show_points defaults to False in plot_publisher_view -- appropriate")
    print("for publishing sensitive data (e.g. salary statistics) without")
    print("exposing individual data points. Pass show_points=True explicitly")
    print("to override, e.g. for internal analysis.")


if __name__ == "__main__":
    main()

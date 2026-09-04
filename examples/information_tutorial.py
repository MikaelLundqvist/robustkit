"""
robustkit.information quickstart tutorial
===========================================

A self-contained walkthrough of feature ranking and quadrant
classification, run against a small synthetic dataset. Run with:

    python examples/information_tutorial.py
"""

import numpy as np
import pandas as pd

from robustkit import rank_features, quadrant_report, plot_feature_space, profile, print_profile


def make_example_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    strong_signal = rng.uniform(0, 10, n)      # drives the target directly
    weak_signal = rng.uniform(0, 10, n)        # barely related
    noise_category = rng.choice([f"cat_{i}" for i in range(20)], size=n)  # high entropy, ~0 MI
    low_cardinality = rng.choice(["A", "B"], size=n)                      # low entropy, weakly related

    target = 100 + 10 * strong_signal + 0.5 * weak_signal + rng.normal(0, 2, n)

    return pd.DataFrame({
        "strong_signal": strong_signal,
        "weak_signal": weak_signal,
        "noise_category": noise_category,
        "low_cardinality": low_cardinality,
        "target": target,
    })


def main():
    df = make_example_data()

    print("=== 1. Dataset profile ===")
    print_profile(df, target="target")
    print()

    print("=== 2. Feature ranking ===")
    ranking = rank_features(df, target="target")
    print(ranking.to_string(index=False))
    print()
    print("Note: noise_category has many levels (high entropy) but low mutual")
    print("information -- high 'information content' does not imply high")
    print("relevance to the target.\n")

    print("=== 3. Quadrant classification ===")
    report = quadrant_report(df, target="target")
    print(report[["feature", "mutual_information", "information_efficiency", "quadrant"]].to_string(index=False))
    print(f"\nThresholds used -- MI: {report.attrs['mi_threshold']:.3f}, "
          f"efficiency: {report.attrs['eff_threshold']:.3f}")
    print()

    print("=== 4. Visualization ===")
    print("Calling plot_feature_space(df, target='target') opens a matplotlib")
    print("figure with features placed in (efficiency, mutual information)")
    print("space, colored by quadrant -- always consistent with the table above.")
    # Uncomment to actually display the plot:
    # plot_feature_space(df, target="target")


if __name__ == "__main__":
    main()

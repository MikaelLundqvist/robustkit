"""
Validation for robustkit.quantiles.reconstruct -- the most statistically
involved part of robustkit.quantiles, so it gets the most thorough
boundary-condition testing here.

Structure follows requirements-based testing practice: each check
states the specific claim/boundary condition it verifies before
verifying it, and checks are grouped as:

    - normal-case calibration accuracy
    - boundary conditions (degenerate spread, invalid ordering,
      non-positive values, extreme dispersion_ratio)
    - convergence behavior (does calibration error shrink as n grows?)
    - reproducibility (same seed -> identical; different seed -> varies,
      but calibration quality is seed-independent)
    - group-structure integrity (expand_aggregated_table)
    - end-to-end validation against real SCB data (a Huber fit on
      reconstructed pseudo-individual data should track the real
      published median trend closely)
    - the two mean-only methods (flat vs. borrowed_dispersion), compared
      side by side rather than trusted individually
"""

from pathlib import Path

import numpy as np
import pandas as pd

from robustkit import (
    expand_aggregated_group, expand_aggregated_table, check_reconstruction_quality,
    expand_aggregated_group_flat, expand_aggregated_group_borrowed_dispersion,
    expand_aggregated_table_flat, expand_aggregated_table_borrowed_dispersion,
    compare_reconstruction_methods,
    load_scb_json_stat, fit_huber_trend, predict_trend,
)

from ..openml_suite.common import section, safe_run

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Normal-case calibration accuracy
# ---------------------------------------------------------------------------

def run_calibration_accuracy():
    section("reconstruct -- normal-case calibration accuracy")

    def check_large_n():
        values = expand_aggregated_group(n=50000, q1=30000, median=38000, q3=48000, seed=0)
        quality = check_reconstruction_quality(values, q1=30000, median=38000, q3=48000)
        for key in ("q1", "median", "q3"):
            assert quality["pct_error"][key] < 2, f"{key} error too high: {quality['pct_error'][key]:.2f}%"
        return quality

    quality = safe_run("large n (50000): all quantile errors < 2%", check_large_n)
    if quality is not None:
        print(f"         pct_error: {quality['pct_error']}")


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def run_boundary_conditions():
    section("reconstruct -- boundary conditions")

    def check_zero_spread():
        values = expand_aggregated_group(n=100, q1=40000, median=40000, q3=40000, seed=1)
        assert np.allclose(values, 40000, rtol=1e-9), "zero-spread case should return the constant value exactly"
        return True

    safe_run("q1 == median == q3 (degenerate zero spread)", check_zero_spread)

    def check_invalid_ordering_raises():
        try:
            expand_aggregated_group(n=10, q1=50000, median=40000, q3=60000)
        except ValueError:
            return True
        raise AssertionError("Expected ValueError for q1 > median")

    safe_run("q1 > median raises ValueError (invalid ordering)", check_invalid_ordering_raises)

    def check_non_positive_raises():
        try:
            expand_aggregated_group(n=10, q1=-5, median=100, q3=200)
        except ValueError:
            return True
        raise AssertionError("Expected ValueError for non-positive q1")

    safe_run("non-positive value raises ValueError (lognormal undefined)", check_non_positive_raises)

    def check_n_equals_one():
        values = expand_aggregated_group(n=1, q1=30000, median=38000, q3=48000, seed=0)
        assert len(values) == 1 and values[0] > 0
        return values[0]

    result = safe_run("n=1 does not crash, returns a single positive draw", check_n_equals_one)
    if result is not None:
        print(f"         single draw: {result:.1f}")

    def check_borrowed_dispersion_extreme_ratio_raises():
        try:
            expand_aggregated_group_borrowed_dispersion(n=10, mean=100, dispersion_ratio=3.0)
        except ValueError:
            return True
        raise AssertionError("Expected ValueError for a dispersion_ratio implying negative Q1")

    safe_run(
        "borrowed_dispersion: dispersion_ratio implying negative Q1 raises ValueError",
        check_borrowed_dispersion_extreme_ratio_raises,
    )

    def check_borrowed_dispersion_negative_ratio_raises():
        try:
            expand_aggregated_group_borrowed_dispersion(n=10, mean=100, dispersion_ratio=-0.1)
        except ValueError:
            return True
        raise AssertionError("Expected ValueError for negative dispersion_ratio")

    safe_run("borrowed_dispersion: negative dispersion_ratio raises ValueError", check_borrowed_dispersion_negative_ratio_raises)


# ---------------------------------------------------------------------------
# Convergence behavior
# ---------------------------------------------------------------------------

def run_convergence_check():
    section("reconstruct -- convergence: does calibration error shrink as n grows?")

    def check_convergence():
        rows = []
        for n in (100, 1000, 10000, 100000):
            values = expand_aggregated_group(n=n, q1=30000, median=38000, q3=48000, seed=0)
            quality = check_reconstruction_quality(values, q1=30000, median=38000, q3=48000)
            rows.append({"n": n, **quality["pct_error"]})
        return pd.DataFrame(rows)

    result = safe_run("pct_error at n=100/1000/10000/100000", check_convergence)
    if result is not None:
        print(result.to_string(index=False))
        # Not a strict monotonic guarantee (still random), but median
        # error at n=100000 should clearly beat n=100 on average.
        if result.loc[result["n"] == 100000, "median"].iloc[0] > result.loc[result["n"] == 100, "median"].iloc[0]:
            print("         NOTE: median error did not improve from n=100 to n=100000 in this run "
                  "(possible with a single seed each -- rerun with multiple seeds if this recurs)")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def run_reproducibility_check():
    section("reconstruct -- reproducibility")

    def check_same_seed_identical():
        a = expand_aggregated_group(n=1000, q1=30000, median=38000, q3=48000, seed=7)
        b = expand_aggregated_group(n=1000, q1=30000, median=38000, q3=48000, seed=7)
        assert np.array_equal(a, b), "same seed should produce identical output"
        return True

    safe_run("same seed -> bit-identical output", check_same_seed_identical)

    def check_different_seed_varies_but_calibrates():
        a = expand_aggregated_group(n=5000, q1=30000, median=38000, q3=48000, seed=1)
        b = expand_aggregated_group(n=5000, q1=30000, median=38000, q3=48000, seed=2)
        assert not np.array_equal(a, b), "different seeds should NOT produce identical output"
        qa = check_reconstruction_quality(a, 30000, 38000, 48000)
        qb = check_reconstruction_quality(b, 30000, 38000, 48000)
        assert qa["pct_error"]["median"] < 5 and qb["pct_error"]["median"] < 5
        return qa, qb

    safe_run("different seeds vary in values but agree in calibration quality", check_different_seed_varies_but_calibrates)


# ---------------------------------------------------------------------------
# Group-structure integrity (table-level functions)
# ---------------------------------------------------------------------------

def run_group_structure_integrity():
    section("reconstruct -- expand_aggregated_table group-structure integrity")

    df = pd.DataFrame({
        "year": [2020, 2021, 2022],
        "n": [500, 600, 450],
        "q1": [30000, 31000, 32500],
        "median": [38000, 39000, 40500],
        "q3": [48000, 49500, 51000],
    })

    def check_sizes_and_labels():
        synthetic = expand_aggregated_table(
            df, n_col="n", q1_col="q1", median_col="median", q3_col="q3",
            group_cols=["year"], value_name="salary", seed=0,
        )
        assert len(synthetic) == df["n"].sum(), "total row count must equal sum of n"
        for _, row in df.iterrows():
            actual_n = (synthetic["year"] == row["year"]).sum()
            assert actual_n == row["n"], f"year {row['year']}: expected {row['n']} rows, got {actual_n}"
        return synthetic

    synthetic = safe_run("row counts per group exactly match n_col", check_sizes_and_labels)
    if synthetic is not None:
        print(f"         total rows: {len(synthetic)} (= {df['n'].sum()})")


# ---------------------------------------------------------------------------
# End-to-end validation against real SCB data
# ---------------------------------------------------------------------------

def run_real_data_validation():
    section("reconstruct -- end-to-end validation against real SCB quartile data")

    quartiles_path = DATA_DIR / "quartiles.json"
    if not quartiles_path.exists():
        print(f"  (skipped -- place a real SCB JSON-stat export at {quartiles_path} to run this check)")
        return

    def load_real_table():
        df = load_scb_json_stat(quartiles_path)
        df["år"] = df["år"].astype(int)
        content_map = {
            "Anställda tjänstemän, privat sektor (SLP)": "n",
            "Totallön, tjänstemän privat sektor (SLP), undre kvartil": "q1",
            "Totallön, tjänstemän privat sektor (SLP), median": "median",
            "Totallön, tjänstemän privat sektor (SLP), övre kvartil": "q3",
        }
        sub = df[df["tabellinnehåll"].isin(content_map)].copy()
        sub["field"] = sub["tabellinnehåll"].map(content_map)
        wide = sub[sub["kön"] == "totalt"].pivot_table(index="år", columns="field", values="value").reset_index()
        # Real n is national-scale (>1 million) -- scaled down here for a
        # tractable demonstration; calibration quality does not depend
        # on reproducing the true population size (see reconstruct.py's
        # docstring warning about the unbounded lognormal tail at large n).
        wide["n_demo"] = 3000
        return wide

    wide = safe_run("load and prepare real quartile table (n scaled to 3000/year for tractability)", load_real_table)
    if wide is None:
        return

    def reconstruct_and_validate():
        synthetic = expand_aggregated_table(
            wide, n_col="n_demo", q1_col="q1", median_col="median", q3_col="q3",
            group_cols=["år"], value_name="salary", seed=42,
        )
        fit = fit_huber_trend(
            synthetic["år"].to_numpy(dtype=float), synthetic["salary"].to_numpy(dtype=float), degree=2,
        )
        years = wide["år"].to_numpy(dtype=float)
        huber_pred = predict_trend(fit, years)

        comparison = pd.DataFrame({
            "year": wide["år"],
            "real_median": wide["median"],
            "huber_on_reconstructed": huber_pred.round(0),
        })
        comparison["pct_diff"] = (
            np.abs(comparison["huber_on_reconstructed"] - comparison["real_median"]) / comparison["real_median"] * 100
        ).round(2)

        max_diff = comparison["pct_diff"].max()
        assert max_diff < 5, f"Huber fit on reconstructed data deviated {max_diff:.1f}% from real median (expected < 5%)"
        return comparison

    comparison = safe_run(
        "Huber fit on RECONSTRUCTED data tracks the REAL published median trend within 5%",
        reconstruct_and_validate,
    )
    if comparison is not None:
        print(comparison.to_string(index=False))
        print(f"         max deviation: {comparison['pct_diff'].max():.2f}%")


# ---------------------------------------------------------------------------
# Mean-only methods: flat vs. borrowed_dispersion, compared side by side
# ---------------------------------------------------------------------------

def run_mean_only_methods_comparison():
    section("reconstruct -- mean-only methods (flat vs. borrowed_dispersion)")

    def check_flat_zero_spread():
        values = expand_aggregated_group_flat(n=200, mean=40000)
        assert np.all(values == 40000) and np.std(values) == 0
        return values

    safe_run("flat method: exactly zero within-group spread", check_flat_zero_spread)

    def check_borrowed_matches_ratio():
        values = expand_aggregated_group_borrowed_dispersion(n=50000, mean=40000, dispersion_ratio=0.45, seed=0)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        actual_ratio = (q3 - q1) / median
        assert abs(actual_ratio - 0.45) < 0.03, f"expected ratio ~0.45, got {actual_ratio:.3f}"
        return actual_ratio

    ratio = safe_run("borrowed_dispersion: reproduces the target dispersion_ratio", check_borrowed_matches_ratio)
    if ratio is not None:
        print(f"         achieved ratio: {ratio:.3f} (target 0.45)")

    def side_by_side():
        return compare_reconstruction_methods(n=5000, mean=52200, dispersion_ratio=0.45, seed=2)

    result = safe_run("side-by-side comparison for a single group (mean=52200)", side_by_side)
    if result is not None:
        print(result["summary"].to_string(index=False))
        print("         NOTE: borrowed_dispersion's mean exceeds the input 'mean' -- expected, "
              "since 'mean' is used as the lognormal's MEDIAN internally, and a lognormal's true "
              "mean is always >= its median (see reconstruct.py docstring).")


def main():
    run_calibration_accuracy()
    run_boundary_conditions()
    run_convergence_check()
    run_reproducibility_check()
    run_group_structure_integrity()
    run_real_data_validation()
    run_mean_only_methods_comparison()


if __name__ == "__main__":
    main()

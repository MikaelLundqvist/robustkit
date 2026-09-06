"""
robustkit.quantiles quickstart tutorial (part 1: io + trend)
================================================================

A self-contained walkthrough of loading a JSON-stat statistical table
and visualizing an already-published quantile trend over time, using
a small synthetic JSON-stat fixture (no real data needed to run this).

This covers the "quantiles are already given" case -- e.g. a national
statistics agency publishing Q1/median/Q3 directly. See a future
reconstruct_tutorial.py for the complementary case: reconstructing
approximate individual-level data from aggregated group summaries.

Run with:

    python examples/quantiles_tutorial.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from robustkit import load_scb_json_stat, plot_quantile_trend, quantile_trend_dispersion


def make_example_json_stat_file(path):
    """
    A small, self-contained JSON-stat fixture in the same structure as
    a real SCB export: one occupation, three quantile measures, six
    years. Mimics the shape of the real SLP salary-quartile tables
    this module was built against, without using any real data.
    """
    years = [str(y) for y in range(2018, 2024)]
    q1_values = [28000, 29000, 30500, 32000, 33800, 35500]
    median_values = [35000, 36500, 38000, 40000, 42200, 44500]
    q3_values = [44000, 45500, 47500, 50000, 52800, 55600]

    dataset = {
        "dimension": {
            "ContentsCode": {
                "label": "tabellinnehåll",
                "category": {
                    "index": {"Q1": 0, "MED": 1, "Q3": 2},
                    "label": {
                        "Q1": "Totallön, undre kvartil",
                        "MED": "Totallön, median",
                        "Q3": "Totallön, övre kvartil",
                    },
                },
            },
            "Tid": {
                "label": "år",
                "category": {
                    "index": {y: i for i, y in enumerate(years)},
                    "label": {y: y for y in years},
                },
            },
            "id": ["ContentsCode", "Tid"],
            "size": [3, len(years)],
            "role": {"time": ["Tid"]},
        },
        "value": q1_values + median_values + q3_values,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset}, f)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "example_scb_table.json"
        make_example_json_stat_file(path)

        # ------------------------------------------------------------
        # 1. Load the JSON-stat file generically
        # ------------------------------------------------------------
        print("=== 1. Loading a JSON-stat table ===")
        df = load_scb_json_stat(path)
        df["år"] = df["år"].astype(int)
        print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        print()

        # ------------------------------------------------------------
        # 2. Reshape from long to a wide quantile table
        # ------------------------------------------------------------
        print("=== 2. Reshaping to a wide quantile table ===")
        quantile_map = {
            "Totallön, undre kvartil": "q1",
            "Totallön, median": "median",
            "Totallön, övre kvartil": "q3",
        }
        df["quantile"] = df["tabellinnehåll"].map(quantile_map)
        wide = df.pivot_table(index="år", columns="quantile", values="value").reset_index()
        print(wide.to_string(index=False))
        print()

        # ------------------------------------------------------------
        # 3. Visualize the published quantile trend directly
        # ------------------------------------------------------------
        print("=== 3. Quantile trend (no estimation needed -- already published) ===")
        trend = plot_quantile_trend(wide, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
        print(trend.to_string(index=False))
        print()
        print("Calling plot_quantile_trend() without further arguments also opens")
        print("a matplotlib figure with the median line and IQR band over time.\n")

        # ------------------------------------------------------------
        # 4. Dispersion over time, from the published quantiles directly
        # ------------------------------------------------------------
        print("=== 4. Dispersion ratio over time ===")
        disp = quantile_trend_dispersion(wide, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
        print(disp[["år", "dispersion_ratio"]].to_string(index=False))
        print()
        print("Note: this required no individual-level data at all -- SCB (or any")
        print("statistics agency publishing quartiles directly) already provides")
        print("everything needed for this analysis.")


if __name__ == "__main__":
    main()

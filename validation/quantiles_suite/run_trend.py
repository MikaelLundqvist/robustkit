"""
Validation for robustkit.quantiles.trend -- visualizing already-
published quantile trends (no individual data, no estimation needed).
"""

from pathlib import Path

import pandas as pd

from robustkit import load_scb_json_stat, plot_quantile_trend, quantile_trend_dispersion
from robustkit.quantiles.trend import prepare_quantile_trend

from ..openml_suite.common import section, safe_run

DATA_DIR = Path(__file__).parent / "data"


def run_trend_synthetic():
    section("robustkit.quantiles.trend -- synthetic checks")

    df = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023],
        "lower": [28000, 29000, 30500, 32000],
        "med": [35000, 36500, 38000, 40000],
        "upper": [44000, 45500, 47500, 50000],
    }).sample(frac=1, random_state=0)  # shuffled input, sorting must be verified

    def check_sort_and_rename():
        trend = prepare_quantile_trend(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")
        assert list(trend.columns) == ["year", "q1", "median", "q3"]
        assert (trend["year"].diff().dropna() > 0).all(), "expected ascending sort by x_col"
        assert (trend["q3"] >= trend["median"]).all()
        assert (trend["median"] >= trend["q1"]).all()
        return trend

    safe_run("prepare_quantile_trend sorts by x and renames columns", check_sort_and_rename)

    def check_dispersion_ratio():
        result = quantile_trend_dispersion(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")
        expected_2020 = (44000 - 28000) / 35000
        actual = result.loc[result["year"] == 2020, "dispersion_ratio"].iloc[0]
        assert abs(actual - expected_2020) < 1e-9, f"expected {expected_2020}, got {actual}"
        return result

    safe_run("quantile_trend_dispersion computes (Q3-Q1)/median correctly", check_dispersion_ratio)

    def check_zero_median():
        zero_df = pd.DataFrame({"year": [2020], "lower": [-5], "med": [0], "upper": [5]})
        result = quantile_trend_dispersion(zero_df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")
        assert result["dispersion_ratio"].iloc[0] == 0.0
        return True

    safe_run("quantile_trend_dispersion handles median=0 without dividing by zero", check_zero_median)


def run_trend_against_real_file():
    section("robustkit.quantiles.trend -- against real SCB quartile table")

    quartiles_path = DATA_DIR / "quartiles.json"
    if not quartiles_path.exists():
        print(f"  (skipped -- place a real SCB JSON-stat export at {quartiles_path} to run this check)")
        return

    def load_and_reshape():
        df = load_scb_json_stat(quartiles_path)
        df["år"] = df["år"].astype(int)

        quantile_map = {
            "Totallön, tjänstemän privat sektor (SLP), undre kvartil": "q1",
            "Totallön, tjänstemän privat sektor (SLP), median": "median",
            "Totallön, tjänstemän privat sektor (SLP), övre kvartil": "q3",
        }
        sub = df[df["tabellinnehåll"].isin(quantile_map)].copy()
        sub["quantile"] = sub["tabellinnehåll"].map(quantile_map)
        wide = sub[sub["kön"] == "totalt"].pivot_table(index="år", columns="quantile", values="value").reset_index()
        return wide

    wide = safe_run("load real quartile table and reshape to wide format", load_and_reshape)
    if wide is None:
        return

    def check_trend():
        trend = plot_quantile_trend(wide, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
        assert (trend["q3"] >= trend["median"]).all()
        assert (trend["median"] >= trend["q1"]).all()
        assert (trend["år"].diff().dropna() > 0).all()
        assert len(trend) == 12, f"expected 12 years, got {len(trend)}"
        return trend

    trend = safe_run("plot_quantile_trend on real data: ordering and completeness", check_trend)
    if trend is not None:
        print(trend.to_string(index=False))

    def check_dispersion():
        disp = quantile_trend_dispersion(wide, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
        assert disp["dispersion_ratio"].between(0, 2).all(), "dispersion_ratio outside a sane range"
        return disp

    disp = safe_run("quantile_trend_dispersion on real data: sane range", check_dispersion)
    if disp is not None:
        print(disp[["år", "dispersion_ratio"]].to_string(index=False))


def main():
    run_trend_synthetic()
    run_trend_against_real_file()


if __name__ == "__main__":
    main()

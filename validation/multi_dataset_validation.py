"""
robustkit multi-dataset validation
=====================================

Runs the full robustkit toolkit (core, segmentation, information,
benchmark, report) against several real, well-known datasets fetched
via sklearn.datasets.fetch_openml -- exploratory validation beyond the
package's own synthetic examples.

This is NOT a pass/fail test suite. Results are meant to be read and
interpreted: some "failures" (a function raising on a genuinely
unsuitable dataset/column combination, e.g. a discrete target with
very few unique values) are informative findings in themselves, not
bugs to fix blindly. Every step is wrapped so one failure doesn't
abort the whole run -- you get a complete report regardless.

Requires internet access on first run (fetch_openml downloads and then
locally caches each dataset). Run with:

    python validation/multi_dataset_validation.py

If a dataset's expected column names don't match (OpenML versions and
mirrors occasionally differ slightly), the script prints the actual
available columns so they can be corrected in the DATASETS list below.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import robustkit as rk

warnings.filterwarnings("ignore")


DATASETS = [
    {
        "name": "house_prices (Ames Housing)",
        "openml_name": "house_prices",
        "openml_version": 1,
        "x_col": "GrLivArea",
        "y_col": "SalePrice",
        "segment_col": "Neighborhood",
        "feature_cols": ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"],
    },
    {
        "name": "abalone",
        "openml_name": "abalone",
        "openml_version": 1,
        "x_col": "Shell_weight",
        "y_col": "Class_number_of_rings",
        "segment_col": "Sex",
        "feature_cols": ["Length", "Diameter", "Height", "Whole_weight", "Shell_weight"],
    },
    {
        "name": "wine-quality-red",
        "openml_name": "wine-quality-red",
        "openml_version": 1,
        "x_col": "alcohol",
        "y_col": "class",
        "segment_col": None,
        "feature_cols": ["alcohol", "volatile_acidity", "sulphates", "citric_acid", "pH"],
    },
    {
        "name": "autoMpg",
        "openml_name": "autoMpg",
        "openml_version": 1,
        "x_col": "horsepower",
        "y_col": "class",
        "segment_col": "origin",
        "feature_cols": ["horsepower", "weight", "displacement", "acceleration"],
    },
    {
        # Included specifically to independently cross-check earlier
        # exploratory findings (via Copilot) on this same dataset:
        # RM/ZN expected "robust"; TAX/CHAS expected "structural
        # sensitivity"; AGE/CRIM expected "fragile". Worth comparing
        # this run's feature_robustness_report against those.
        "name": "boston (housing)",
        "openml_name": "boston",
        "openml_version": 1,
        "x_col": "LSTAT",
        "y_col": "MEDV",
        "segment_col": None,
        "feature_cols": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
    },
    {
        # Large n (~54000), strongly right-skewed price -- a good
        # stress test for scale and for the report module's
        # analyst/publisher-view distinction.
        "name": "diamonds",
        "openml_name": "diamonds",
        "openml_version": 1,
        "x_col": "carat",
        "y_col": "price",
        "segment_col": "cut",
        "feature_cols": ["carat", "depth", "table"],
    },
]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_run(label, fn):
    """Run fn(), reporting and swallowing any exception rather than aborting the run."""
    try:
        result = fn()
        print(f"  [OK]   {label}")
        return result
    except Exception as exc:
        print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        return None


def load_dataset(cfg):
    data = fetch_openml(name=cfg["openml_name"], version=cfg["openml_version"], as_frame=True, parser="auto")
    df = data.frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


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


def run_segmentation_module(df, x_col, y_col, segment_col):
    section(f"robustkit.segmentation -- segment={segment_col}")
    if segment_col is None:
        print("  (skipped -- no segment column defined for this dataset)")
        return

    sub = df[[x_col, y_col, segment_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    sub[segment_col] = sub[segment_col].astype(str)

    segmented = safe_run("hierarchical_segment", lambda: rk.hierarchical_segment(sub, [[segment_col]], min_size=20))
    if segmented is not None:
        print("  segment sizes:")
        print("   ", rk.segment_sizes(segmented).to_string().replace("\n", "\n    "))

        report = safe_run(
            "apply_by_segment(model_stability_pct)",
            lambda: rk.apply_by_segment(segmented, "segment_id", x_col, y_col, rk.model_stability_pct),
        )
        if report is not None:
            cols = [c for c in ["segment", "n", "skipped", "median_pct_diff"] if c in report.columns]
            print(report[cols].to_string(index=False))


def run_information_module(df, y_col, feature_cols):
    section(f"robustkit.information -- target={y_col}")
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print(f"  (skipped -- none of the expected feature columns {feature_cols} found)")
        return

    sub = df[cols + [y_col]].copy().dropna()

    ranking = safe_run("rank_features", lambda: rk.rank_features(sub, target=y_col))
    if ranking is not None:
        print(ranking.to_string(index=False))

    report = safe_run("quadrant_report", lambda: rk.quadrant_report(sub, target=y_col))
    if report is not None:
        print(report[["feature", "quadrant"]].to_string(index=False))

    if len(cols) >= 2:
        pairs = safe_run(
            "rank_communicative_pairs",
            lambda: rk.rank_communicative_pairs(sub, target=y_col, features=cols[:4]),
        )
        if pairs is not None:
            print(pairs[["feature_1", "feature_2", "redundancy", "synergy", "pair_score"]].to_string(index=False))


def run_benchmark_module(df, x_col, y_col, segment_col, feature_cols):
    section(f"robustkit.benchmark -- segment={segment_col}")

    cols = list(dict.fromkeys([x_col, y_col] + feature_cols + ([segment_col] if segment_col else [])))
    sub = df[cols].copy()
    for c in [x_col, y_col] + feature_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    if segment_col is not None:
        sub[segment_col] = sub[segment_col].astype(str)
        sizes = sub[segment_col].value_counts()
        keep = sizes[sizes >= 20].index
        seg_sub = sub[sub[segment_col].isin(keep)]

        if seg_sub[segment_col].nunique() < 2:
            print("  (skipped segment_position_report -- fewer than 2 segments have >= 20 rows)")
        else:
            report = safe_run(
                "segment_position_report",
                lambda: rk.segment_position_report(seg_sub, segment_col=segment_col, x_col=x_col, y_col=y_col, n_boot=200),
            )
            if report is not None:
                print(report.to_string(index=False))
    else:
        print("  (skipped segment_position_report -- no segment column defined for this dataset)")

    # feature_robustness_report needs >= 2 features to classify meaningfully
    # (see its docstring) and does NOT need a segment column at all.
    rob_features = list(dict.fromkeys([x_col] + [c for c in feature_cols if c != x_col]))
    rob_features = [c for c in rob_features if c in sub.columns][:4]

    if len(rob_features) < 2:
        print(f"  (skipped feature_robustness_report -- fewer than 2 usable features: {rob_features})")
        return

    rob_report = safe_run(
        f"feature_robustness_report (features={rob_features})",
        lambda: rk.feature_robustness_report(sub, target=y_col, features=rob_features),
    )
    if rob_report is not None:
        print(rob_report.to_string(index=False))


def run_report_module(x, y):
    section("robustkit.report -- analyst vs. publisher view")
    safe_run("plot_analyst_view", lambda: rk.plot_analyst_view(x, y, n_boot=200, show_points=False))
    binned = safe_run("plot_publisher_view", lambda: rk.plot_publisher_view(x, y, n_bins=8, show_points=False))
    if binned is not None:
        print(binned[["x_center", "n", "q1", "median", "q3", "dispersion_ratio"]].to_string(index=False))


def main():
    import matplotlib
    matplotlib.use("Agg")

    for cfg in DATASETS:
        section(f"DATASET: {cfg['name']}")
        df = safe_run(f"fetch_openml({cfg['openml_name']!r})", lambda cfg=cfg: load_dataset(cfg))
        if df is None:
            continue

        print(f"  Shape: {df.shape}")
        preview_cols = list(df.columns)[:15]
        print(f"  Columns (first 15): {preview_cols}{' ...' if df.shape[1] > 15 else ''}")

        missing_cols = [c for c in (cfg["x_col"], cfg["y_col"]) if c not in df.columns]
        if missing_cols:
            print(f"  [FAIL] Expected column(s) not found: {missing_cols}")
            print(f"  All available columns: {list(df.columns)}")
            continue

        x, y = run_core_module(df, cfg["x_col"], cfg["y_col"])
        run_segmentation_module(df, cfg["x_col"], cfg["y_col"], cfg["segment_col"])
        run_information_module(df, cfg["y_col"], cfg["feature_cols"])
        run_benchmark_module(df, cfg["x_col"], cfg["y_col"], cfg["segment_col"], cfg["feature_cols"])
        run_report_module(x, y)

    section("VALIDATION RUN COMPLETE")


if __name__ == "__main__":
    main()

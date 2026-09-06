"""
The "Robustness Map": classify features by how much a conclusion about
their relationship with the target depends on (a) which robust fitting
method is used, and (b) which specific observations are included.

Motivated by a concrete finding: two features can have a similar
number of Cook's-distance-flagged observations while having wildly
different actual influence on the fitted trend (cook_impact). Model
stability and Cook-impact are answering genuinely different questions,
and a feature's position on both axes together says more about how
much to trust a conclusion involving it than either axis alone.
"""

import pandas as pd

from ..core.stability import model_stability_pct
from ..core.diagnostics import cooks_diagnostic, cook_impact
from ..common.quadrants import classify_quadrants

ROBUSTNESS_LABELS = {
    "high_high": "fragile",                     # unstable AND data-driven -- least trustworthy
    "high_y_only": "data_sensitive",             # stable across methods, but driven by a few points
    "high_x_only": "structural_sensitivity",     # sensitive to method choice, not to specific points
    "low_low": "robust",                         # stable across methods AND not driven by outliers
}

_ROBUSTNESS_DISPLAY = {
    "robust": "Robust",
    "structural_sensitivity": "Structural Sensitivity",
    "data_sensitive": "Data Sensitive",
    "fragile": "Fragile",
}


def feature_robustness_report(df, target, features=None, degree=2,
                               stability_threshold="median", impact_threshold="median"):
    """
    For each candidate feature (used as x against target as y),
    compute model_stability_pct and cook_impact, and classify the
    feature into one of four robustness quadrants:

        robust                  -- low stability spread, low Cook impact
        structural_sensitivity  -- high stability spread, low Cook impact
                                    (the conclusion depends on which
                                    fitting method you pick, not on
                                    specific data points)
        data_sensitive          -- low stability spread, high Cook impact
                                    (a handful of observations drive
                                    the conclusion, but method choice
                                    barely matters)
        fragile                 -- high stability spread AND high Cook
                                    impact (the least trustworthy)

    Requires at least 2 features. Quadrant assignment is threshold-
    based (median by default, see robustkit.classify_quadrants) --
    with a single feature, that feature is trivially "at or above its
    own median" on both axes, so it would always be classified
    "fragile" regardless of its actual stability/impact values. This
    is a property of median-based thresholding with n=1, not a
    meaningful result, so it is rejected explicitly here rather than
    silently returning a misleading label.
    """
    features = features or [c for c in df.columns if c != target]
    if len(features) < 2:
        raise ValueError(
            f"feature_robustness_report requires at least 2 features to classify "
            f"meaningfully (quadrant thresholds are computed across the candidate "
            f"features); got {len(features)}: {features}. With a single feature, "
            f"median-based thresholds are degenerate and always classify it as "
            f"'fragile' regardless of its actual values."
        )

    y = df[target].to_numpy(dtype=float)

    rows = []
    for feature in features:
        x = df[feature].to_numpy(dtype=float)

        stability = model_stability_pct(x, y, degree=degree)
        diag = cooks_diagnostic(x, y, degree=degree)

        if len(diag["flagged_indices"]) > 0:
            impact = cook_impact(x, y, diag["flagged_indices"], degree=degree)
            cook_impact_pct = impact["median_pct_change"]
        else:
            cook_impact_pct = 0.0

        rows.append({
            "feature": feature,
            "stability_pct": stability["median_pct_diff"],
            "cook_impact_pct": cook_impact_pct,
            "n_flagged": len(diag["flagged_indices"]),
        })

    report = pd.DataFrame(rows)
    classified = classify_quadrants(
        report, x_col="stability_pct", y_col="cook_impact_pct",
        x_threshold=stability_threshold, y_threshold=impact_threshold,
        labels=ROBUSTNESS_LABELS,
    )
    return classified.sort_values("cook_impact_pct", ascending=False).reset_index(drop=True)


def plot_feature_robustness(df=None, target=None, report=None, annotate=True, figsize=(10, 7), ax=None):
    """
    Scatter plot of features in robustness space: x = model stability
    spread (%), y = Cook-impact (%), colored by robustness quadrant.

    Provide either a precomputed `report` (ideally the output of
    feature_robustness_report, so quadrant labels are already
    attached) or a `df`/`target` pair to compute everything internally.
    Always routes quadrant assignment through the same logic as
    feature_robustness_report, so the plot and the table can never
    disagree.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if report is None or "quadrant" not in report.columns:
        report = feature_robustness_report(df, target=target)

    colors_map = {
        "robust": "green",
        "structural_sensitivity": "steelblue",
        "data_sensitive": "darkorange",
        "fragile": "red",
    }
    colors = report["quadrant"].map(colors_map)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        report["stability_pct"], report["cook_impact_pct"],
        c=colors, s=80, edgecolors="black", linewidths=0.5, alpha=0.8,
    )

    if annotate:
        for _, row in report.iterrows():
            ax.annotate(
                row["feature"], (row["stability_pct"], row["cook_impact_pct"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points",
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=_ROBUSTNESS_DISPLAY[quadrant],
               markerfacecolor=color, markersize=10)
        for quadrant, color in colors_map.items()
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_xlabel("Model Stability Spread (%)")
    ax.set_ylabel("Cook Impact (%)")
    ax.set_title("Feature Robustness Map")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return report

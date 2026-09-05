"""
Visualize features in information space (efficiency vs. mutual
information).

plot_feature_space always computes (or accepts) a quadrant-labeled
ranking via quadrants.quadrant_report -- it never applies its own,
separate threshold. This guarantees the plot and quadrant_report() can
never disagree about which quadrant a feature falls into.
"""

from .quadrants import quadrant_report, QUADRANT_LABELS

_QUADRANT_COLORS = {
    "star": "green",
    "power": "steelblue",
    "efficient": "darkorange",
    "weak": "gray",
}


def plot_feature_space(df=None, target=None, ranking=None, annotate=True, figsize=(10, 7), ax=None):
    """
    Scatter plot of features in information space: x = information
    efficiency, y = mutual information, bubble size = entropy (bits),
    color = quadrant.

    Provide either a precomputed `ranking` (ideally the output of
    quadrant_report, so quadrant labels are already attached) or a
    `df`/`target` pair to compute everything internally.

    Returns the quadrant-labeled ranking DataFrame (so you can inspect
    or reuse it), regardless of whether a plot was drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ranking is None or "quadrant" not in ranking.columns:
        ranking = quadrant_report(df=df, target=target, ranking=ranking)

    colors = ranking["quadrant"].map(_QUADRANT_COLORS)
    sizes = ranking["entropy_bits"] * 60 + 20

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        ranking["information_efficiency"], ranking["mutual_information"],
        s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidths=0.5,
    )

    if annotate:
        for _, row in ranking.iterrows():
            ax.annotate(
                row["feature"],
                (row["information_efficiency"], row["mutual_information"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points",
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=label,
               markerfacecolor=_QUADRANT_COLORS[quadrant], markersize=10)
        for quadrant, label in QUADRANT_LABELS.items()
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_xlabel("Information Efficiency (MI per bit of feature entropy)")
    ax.set_ylabel("Mutual Information")
    ax.set_title("Feature Information Space")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return ranking


def feature_map(df, target, **kwargs):
    """Convenience wrapper: rank features and plot them in one call."""
    return plot_feature_space(df=df, target=target, **kwargs)

"""
Classify features into four quadrants based on mutual information and
information efficiency.

Both quadrant_report() and visualization.plot_feature_space() route
through this module's thresholding logic, so a feature can never be
labeled differently by the table and the plot -- a bug present in an
earlier version of this code, where the plot used a fixed 75th
percentile cutoff while the report used the median.
"""

from .mutual_info import rank_features

QUADRANT_LABELS = {
    "star": "Star",            # high MI, high efficiency
    "power": "Predictive",     # high MI, low efficiency: informative but "expensive"
    "efficient": "Efficient",  # low MI, high efficiency: cheap but limited ceiling
    "weak": "Weak",            # low MI, low efficiency
}


def quadrant_report(df=None, target=None, ranking=None, mi_threshold="median", eff_threshold="median"):
    """
    Classify features into four quadrants of (mutual information,
    information efficiency) space.

    Provide either a precomputed `ranking` (output of rank_features)
    or a `df`/`target` pair to compute it internally.

    mi_threshold / eff_threshold: "median" (default) or a quantile in
    (0, 1) used as the cutoff for "high" on each axis. Both quadrant
    thresholds are stored in the returned DataFrame's `.attrs` for
    inspection or reuse (e.g. by plot_feature_space).
    """
    if ranking is None:
        if df is None or target is None:
            raise ValueError("Provide either `ranking` or both `df` and `target`.")
        ranking = rank_features(df, target=target)
    else:
        ranking = ranking.copy()

    mi_cut = (
        ranking["mutual_information"].median()
        if mi_threshold == "median"
        else ranking["mutual_information"].quantile(mi_threshold)
    )
    eff_cut = (
        ranking["information_efficiency"].median()
        if eff_threshold == "median"
        else ranking["information_efficiency"].quantile(eff_threshold)
    )

    def _label(row):
        high_mi = row["mutual_information"] >= mi_cut
        high_eff = row["information_efficiency"] >= eff_cut
        if high_mi and high_eff:
            return "star"
        if high_mi:
            return "power"
        if high_eff:
            return "efficient"
        return "weak"

    ranking["quadrant"] = ranking.apply(_label, axis=1)
    ranking.attrs["mi_threshold"] = mi_cut
    ranking.attrs["eff_threshold"] = eff_cut
    return ranking

"""
Classify features into four quadrants based on mutual information and
information efficiency.

This is a thin wrapper around robustkit.common.quadrants.classify_quadrants
-- the actual thresholding logic lives there and is shared with other
modules (e.g. robustkit.benchmark's Robustness Map), so a feature can
never be labeled differently by different parts of the package.
"""

from ..common.quadrants import classify_quadrants
from .mutual_info import rank_features

QUADRANT_LABELS = {
    "star": "Star",            # high MI, high efficiency
    "power": "Predictive",     # high MI, low efficiency: informative but "expensive"
    "efficient": "Efficient",  # low MI, high efficiency: cheap but limited ceiling
    "weak": "Weak",            # low MI, low efficiency
}

_LABEL_MAP = {
    "high_high": "star",
    "high_y_only": "power",       # high MI (y-axis), low efficiency
    "high_x_only": "efficient",   # high efficiency (x-axis), low MI
    "low_low": "weak",
}


def quadrant_report(df=None, target=None, ranking=None, mi_threshold="median", eff_threshold="median"):
    """
    Classify features into four quadrants of (mutual information,
    information efficiency) space.

    Provide either a precomputed `ranking` (output of rank_features)
    or a `df`/`target` pair to compute it internally.

    mi_threshold / eff_threshold: "median" (default) or a quantile in
    (0, 1) used as the cutoff for "high" on each axis.
    """
    if ranking is None:
        if df is None or target is None:
            raise ValueError("Provide either `ranking` or both `df` and `target`.")
        ranking = rank_features(df, target=target)
    else:
        ranking = ranking.copy()

    classified = classify_quadrants(
        ranking, x_col="information_efficiency", y_col="mutual_information",
        x_threshold=eff_threshold, y_threshold=mi_threshold, labels=_LABEL_MAP,
    )

    # Preserve the original attrs naming used throughout this module
    # and its tests/tutorials.
    classified.attrs["eff_threshold"] = classified.attrs.pop("x_threshold")
    classified.attrs["mi_threshold"] = classified.attrs.pop("y_threshold")
    return classified

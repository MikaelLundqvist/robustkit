"""
Score how "communicable" a single feature is -- not just how
predictive it is, but how easy it would be to build a clear, honest
chart or table around it for a non-technical audience.

This combines several sub-scores into one composite communication_index:
    mutual_information : how related the feature is to the target
    stability           : how homogeneous the target is within each
                           group of this feature, relative to overall
                           spread (low within-group MAD is good)
    group_size_score    : how far the smallest group is above a
                           minimum viable size (a group too small to
                           get its own chart/segment hurts communicability
                           even if it's statistically informative)
    compressibility      : fewer categories are easier to show in a
                           single chart
    interpretability     : an optional user-supplied prior weight
                           (e.g. domain knowledge about how legible a
                           feature is to the intended audience),
                           independent of its statistical properties
"""

import numpy as np
import pandas as pd

from .utils import discretize
from .mutual_info import rank_features

DEFAULT_WEIGHTS = {
    "mutual_information": 0.35,
    "stability": 0.25,
    "group_size_score": 0.2,
    "compressibility": 0.1,
    "interpretability": 0.1,
}


def _stability_score(df, feature, target):
    groups = discretize(df[feature])
    overall_mad = (df[target] - df[target].median()).abs().median()
    if overall_mad == 0:
        return 0.0

    within_group_mads, weights = [], []
    for _, group in df.groupby(groups, observed=True):
        if len(group) < 2:
            continue
        mad = (group[target] - group[target].median()).abs().median()
        within_group_mads.append(mad)
        weights.append(len(group))

    if not within_group_mads:
        return 0.0

    avg_within_mad = np.average(within_group_mads, weights=weights)
    return float(np.clip(1 - avg_within_mad / overall_mad, 0, 1))


def _group_size_score(df, feature, min_group_size=20):
    groups = discretize(df[feature])
    sizes = df.groupby(groups, observed=True).size()
    if len(sizes) == 0:
        return 0.0
    return float(np.clip(sizes.min() / min_group_size, 0, 1))


def _compressibility_score(df, feature, max_reasonable_groups=8):
    groups = discretize(df[feature])
    n_groups = len(pd.unique(groups))
    return float(np.clip(1 - (n_groups - 1) / max_reasonable_groups, 0, 1))


def communication_score(df, feature, target, min_group_size=20, weights=None,
                         interpretability=None, mi_value=None, max_mi=None):
    """
    Compute the composite communication_index for a single feature.

    mi_value / max_mi: precomputed mutual information for this feature
        and the maximum MI across the candidate set being compared,
        used to normalize this feature's MI onto a 0-1 scale that is
        comparable across features. rank_by_communication always
        supplies both; if omitted (e.g. calling this function on a
        single feature in isolation), MI is computed just for this
        feature and normalized against itself (mi_norm=1.0).
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    interpretability = 1.0 if interpretability is None else interpretability

    if mi_value is None:
        ranking = rank_features(df[[feature, target]], target=target)
        mi_value = float(ranking.loc[ranking["feature"] == feature, "mutual_information"].iloc[0])
    if not max_mi:
        max_mi = mi_value if mi_value > 0 else 1.0

    mi_norm = float(np.clip(mi_value / max_mi, 0, 1))
    stability = _stability_score(df, feature, target)
    group_size_score = _group_size_score(df, feature, min_group_size=min_group_size)
    compressibility = _compressibility_score(df, feature)

    sub_scores = {
        "mutual_information": mi_norm,
        "stability": stability,
        "group_size_score": group_size_score,
        "compressibility": compressibility,
        "interpretability": interpretability,
    }

    communication_index = sum(weights[k] * sub_scores[k] for k in weights)

    return {
        "feature": feature,
        "raw_mutual_information": float(mi_value),
        **sub_scores,
        "communication_index": float(communication_index),
    }


def rank_by_communication(df, target, features=None, min_group_size=20, weights=None, interpretability=None):
    """
    Compute communication_score for every candidate feature (or a
    given subset) and return them ranked by communication_index.

    interpretability: optional dict of {feature_name: prior in [0, 1]}.
    Features not present in the dict default to 1.0 (no penalty/bonus).
    """
    features = features or [c for c in df.columns if c != target]
    interpretability = interpretability or {}

    full_ranking = rank_features(df[features + [target]], target=target)
    max_mi = full_ranking["mutual_information"].max()

    rows = []
    for feature in features:
        mi_value = float(full_ranking.loc[full_ranking["feature"] == feature, "mutual_information"].iloc[0])
        rows.append(
            communication_score(
                df, feature, target, min_group_size=min_group_size, weights=weights,
                interpretability=interpretability.get(feature),
                mi_value=mi_value, max_mi=max_mi,
            )
        )

    return pd.DataFrame(rows).sort_values("communication_index", ascending=False).reset_index(drop=True)

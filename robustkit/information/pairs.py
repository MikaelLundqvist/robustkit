"""
Evaluate pairs of features together: how redundant are they with each
other, and does combining them reveal more about the target than
either does alone (synergy)?
"""

import numpy as np
import pandas as pd

from .entropy import entropy
from .mutual_info import rank_features, _mutual_info_between
from .conditional_mi import conditional_mutual_information


def pair_redundancy(df, feature_1, feature_2, seed=0):
    """
    Normalized mutual information between two features themselves
    (not the target): 0 = independent, 1 = fully redundant.

    Normalized by the smaller of the two features' own entropy, since
    MI(A;B) can never exceed min(H(A), H(B)).
    """
    mi = _mutual_info_between(df[feature_1], df[feature_2], seed=seed)
    denom = min(entropy(df[feature_1]), entropy(df[feature_2]))
    if denom <= 0:
        return 0.0
    return float(np.clip(mi / denom, 0, 1))


def pair_synergy(df, feature_1, feature_2, target, seed=0):
    """
    How much extra information does feature_2 contribute about the
    target once feature_1 is already known, beyond what feature_2
    contributes on its own?

        synergy = I(feature_2; target | feature_1) - I(feature_2; target)

    Positive synergy: feature_1 "unlocks" additional predictive value
    in feature_2 (e.g. an interaction effect -- feature_2 only matters
    within certain values of feature_1). Near-zero or negative:
    feature_2 adds little beyond what it already tells you on its own.
    """
    marginal_mi = _mutual_info_between(df[feature_2], df[target], seed=seed)
    conditional_mi = conditional_mutual_information(df, feature_2, target, condition_on=feature_1, seed=seed)
    return float(conditional_mi - marginal_mi)


def rank_communicative_pairs(df, target, features=None, seed=0):
    """
    Evaluate every candidate pair of features and rank them by a
    combined score that rewards individual relevance and synergy while
    penalizing redundancy between the two features.

    pair_score = (mi_feature_1 + mi_feature_2) * (1 - redundancy) + max(synergy, 0)

    synergy here is the sum of both directions (how much feature_2
    gains from knowing feature_1, plus how much feature_1 gains from
    knowing feature_2), since either framing is a valid case for using
    the pair together.
    """
    features = features or [c for c in df.columns if c != target]
    single_ranking = rank_features(df[features + [target]], target=target)
    mi_lookup = dict(zip(single_ranking["feature"], single_ranking["mutual_information"]))

    rows = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            redundancy = pair_redundancy(df, f1, f2, seed=seed)
            synergy = (
                pair_synergy(df, f1, f2, target, seed=seed)
                + pair_synergy(df, f2, f1, target, seed=seed)
            )
            combined_mi = mi_lookup[f1] + mi_lookup[f2]
            dominant = f1 if mi_lookup[f1] >= mi_lookup[f2] else f2
            pair_score = combined_mi * (1 - redundancy) + max(synergy, 0)

            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "mi_feature_1": mi_lookup[f1],
                "mi_feature_2": mi_lookup[f2],
                "redundancy": redundancy,
                "synergy": synergy,
                "dominant_feature": dominant,
                "pair_score": pair_score,
            })

    return pd.DataFrame(rows).sort_values("pair_score", ascending=False).reset_index(drop=True)

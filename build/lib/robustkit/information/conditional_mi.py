"""
Conditional mutual information: how much additional information does
a candidate feature provide about the target, once another feature is
already known?

Estimated via stratification: split the data by the conditioning
feature (discretized if continuous), compute mutual information
within each stratum, and take the size-weighted average across strata.
Strata with too few points to estimate MI meaningfully are skipped
rather than allowed to distort the estimate.
"""

from .utils import discretize
from .mutual_info import prepare_features, _target_is_continuous, _discrete_mask, _mi_regression_bits, _mi_classif_bits


def conditional_mutual_information(df, feature, target, condition_on, seed=0, min_stratum_size=5):
    """
    Estimate I(feature; target | condition_on), in bits.

    Returns 0.0 if every stratum is too small to estimate from (e.g.
    condition_on has too many distinct values relative to the dataset
    size) -- treated as "no usable conditional signal" rather than
    raising an error.
    """
    strata = discretize(df[condition_on])
    y = df[target]
    continuous_target = _target_is_continuous(y)

    total_n = len(df)
    weighted_mi = 0.0

    for _, group in df.groupby(strata, observed=True):
        if len(group) < min_stratum_size:
            continue

        feature_df = group[[feature]]
        discrete = _discrete_mask(feature_df)
        X = prepare_features(feature_df)
        y_group = group[target]

        if continuous_target:
            mi = _mi_regression_bits(X, y_group, discrete_features=discrete, seed=seed)[0]
        else:
            y_enc = y_group.astype("category").cat.codes
            mi = _mi_classif_bits(X, y_enc, discrete_features=discrete, seed=seed)[0]

        weighted_mi += (len(group) / total_n) * mi

    return float(weighted_mi)

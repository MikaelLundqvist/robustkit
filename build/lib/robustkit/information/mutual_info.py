"""
Rank features by mutual information with a target, normalized by each
feature's own entropy ("information efficiency" -- how much of a
feature's information content is actually being used to predict the
target).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .entropy import entropy
from .utils import discretize

_LN2 = np.log(2)


def _mi_regression_bits(X, y, discrete_features, seed=0):
    """
    sklearn's mutual_info_regression returns values in nats (natural
    log); this project's entropy() and all other information-theoretic
    quantities are expressed in bits (log base 2). Every call to
    sklearn's MI estimators should go through this wrapper (or
    _mi_classif_bits below) rather than calling sklearn directly, to
    keep units consistent everywhere -- mixing nats and bits silently
    produces numbers that are wrong by a constant factor of ln(2)
    (~0.693), which is easy to miss since it doesn't look obviously
    broken.
    """
    mi_nats = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=seed)
    return mi_nats / _LN2


def _mi_classif_bits(X, y, discrete_features, seed=0):
    """See _mi_regression_bits -- same nats-to-bits conversion."""
    mi_nats = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=seed)
    return mi_nats / _LN2


def prepare_features(X):
    """
    Minimal default preprocessing: median-impute numeric columns,
    fill missing categoricals with an explicit "Missing" category,
    then integer-encode all categoricals. Intended as a reasonable
    default for mutual information estimation, not a general-purpose
    preprocessing pipeline.
    """
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("Missing").astype("category").cat.codes
    return X


def information_efficiency(mutual_information, entropy_bits):
    """
    Mutual information per bit of the feature's own entropy. Answers:
    "of everything this feature could tell us, how much is actually
    being used to predict the target?" A high-cardinality feature can
    have high raw mutual information while still being inefficient --
    most of its information content goes unused.

    Note: for continuous features, `mutual_information` is estimated
    on the full-resolution continuous values, while `entropy_bits` is
    computed on a binned (discretized) version of the same feature
    (see rank_features' `n_bins`). Binning necessarily discards some
    information, so entropy_bits is a lower bound on the feature's
    true entropy -- meaning the ratio can exceed 1.0. This is expected,
    not a bug: it signals that the feature carries more usable
    information than a coarse categorical summary of it would capture.
    """
    if entropy_bits <= 0:
        return 0.0
    return float(mutual_information / entropy_bits)


def _target_is_continuous(y, max_categories=20):
    """
    Heuristic: numeric dtype with more than `max_categories` distinct
    values is treated as continuous; everything else (non-numeric, or
    numeric with few distinct values -- e.g. an integer-coded class
    label) is treated as categorical.
    """
    return pd.api.types.is_numeric_dtype(y) and y.nunique() > max_categories


def _discrete_mask(X):
    """
    Boolean mask, in column order, indicating which columns of the
    *original* (pre-encoding) DataFrame were categorical. Used to tell
    sklearn's mutual_info_* functions which columns of the encoded
    matrix are discrete rather than continuous.

    This matters more than it might seem: sklearn's default
    (discrete_features="auto") treats all *dense* input as continuous,
    which systematically underestimates mutual information for
    integer-coded categorical features (e.g. two exactly duplicated
    categorical columns should show MI equal to their shared entropy,
    not a fraction of it).
    """
    return [not pd.api.types.is_numeric_dtype(X[col]) for col in X.columns]


def _mutual_info_between(a, b, seed=0):
    """
    Mutual information between two arbitrary series (numeric or
    categorical), used when neither one is conceptually "the target"
    -- e.g. redundancy between two candidate features.

    Mutual information is symmetric in theory; which series plays the
    role of "target" in sklearn's API only determines which estimator
    is used (regression vs. classification), not the underlying
    quantity being estimated (up to estimation noise from the two
    algorithms differing slightly).
    """
    a_df = pd.DataFrame({"a": a})
    discrete = _discrete_mask(a_df)
    X = prepare_features(a_df)

    if _target_is_continuous(b):
        return float(_mi_regression_bits(X, b, discrete_features=discrete, seed=seed)[0])

    b_enc = b.astype("category").cat.codes
    return float(_mi_classif_bits(X, b_enc, discrete_features=discrete, seed=seed)[0])


def rank_features(df, target, seed=0, n_bins=5):
    """
    Rank every column in df (except target) by mutual information with
    target and by information efficiency.

    Automatically detects whether target should be treated as
    continuous (mutual_info_regression) or categorical
    (mutual_info_classif) via _target_is_continuous.

    n_bins: number of quantile bins used to discretize continuous
        features before computing their entropy (via
        information.utils.discretize). entropy() expects categorical
        input; without this step, a continuous feature's entropy would
        be computed as if every distinct value were its own category
        -- inflating entropy_bits toward log2(n_rows) and making
        information_efficiency artificially low. Categorical features
        are unaffected (discretize leaves them as-is).
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_enc = prepare_features(X)
    discrete = _discrete_mask(X)

    if _target_is_continuous(y):
        mi = _mi_regression_bits(X_enc, y, discrete_features=discrete, seed=seed)
    else:
        y_enc = y.astype("category").cat.codes
        mi = _mi_classif_bits(X_enc, y_enc, discrete_features=discrete, seed=seed)

    rows = []
    for col, mi_value in zip(X.columns, mi):
        bits = entropy(discretize(X[col], n_bins=n_bins))
        rows.append({
            "feature": col,
            "mutual_information": float(mi_value),
            "entropy_bits": bits,
            "information_efficiency": information_efficiency(mi_value, bits),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("information_efficiency", ascending=False)
        .reset_index(drop=True)
    )

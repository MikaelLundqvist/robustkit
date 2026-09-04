"""
Rank features by mutual information with a target, normalized by each
feature's own entropy ("information efficiency" -- how much of a
feature's information content is actually being used to predict the
target).
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .entropy import entropy


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


def rank_features(df, target, seed=0):
    """
    Rank every column in df (except target) by mutual information with
    target and by information efficiency.

    Automatically detects whether target should be treated as
    continuous (mutual_info_regression) or categorical
    (mutual_info_classif) via _target_is_continuous.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_enc = prepare_features(X)

    if _target_is_continuous(y):
        mi = mutual_info_regression(X_enc, y, random_state=seed)
    else:
        y_enc = y.astype("category").cat.codes
        mi = mutual_info_classif(X_enc, y_enc, random_state=seed)

    rows = []
    for col, mi_value in zip(X.columns, mi):
        bits = entropy(X[col])
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

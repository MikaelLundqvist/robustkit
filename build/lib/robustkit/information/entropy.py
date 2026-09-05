"""
Shannon entropy for categorical (or already-discretized) variables.
"""

import numpy as np
import pandas as pd


def entropy(series):
    """
    Shannon entropy in bits of a pandas Series, treating its values as
    categorical. For continuous data, discretize (bin) it before
    calling this -- entropy on raw continuous values is not meaningful
    here.
    """
    counts = pd.Series(series).value_counts(normalize=True)
    probs = counts.to_numpy()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

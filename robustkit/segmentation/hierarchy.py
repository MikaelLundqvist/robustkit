"""
Hierarchical segmentation with a minimum-size fallback.

The problem this solves: you want to analyze y-vs-x within meaningful
subgroups (e.g. department x seniority x status), but the finest-
grained grouping often produces some subgroups too small to analyze
reliably. hierarchical_segment() assigns each row to the finest
grouping it belongs to that still meets a minimum size, falling back
to progressively coarser groupings -- and finally to a single "ALL"
segment -- for whatever doesn't fit anywhere finer.
"""

import pandas as pd


def hierarchical_segment(df, hierarchy, min_size=20):
    """
    Parameters
    ----------
    df : pandas.DataFrame
    hierarchy : list of lists of column names, ordered from finest to
        coarsest, e.g. [["family", "level", "flag"], ["level", "flag"], ["flag"]]
    min_size : int
        Minimum number of rows a group must have to be used as a
        segment at a given hierarchy level.

    Returns
    -------
    pandas.DataFrame
        A copy of df with two new columns:
            segment_id    -- string label identifying the assigned segment
            segment_level -- index into `hierarchy` at which the row was
                              assigned (0 = finest grouping); equals
                              len(hierarchy) for rows that fell all the
                              way back to a single "ALL" segment
    """
    df = df.copy()
    df["segment_id"] = pd.NA
    df["segment_level"] = pd.NA

    remaining_idx = df.index

    for level_idx, cols in enumerate(hierarchy):
        if len(remaining_idx) == 0:
            break

        subset = df.loc[remaining_idx]
        sizes = subset.groupby(cols, observed=True).size()
        valid_groups = sizes[sizes >= min_size].index

        if len(valid_groups) == 0:
            continue

        group_keys = subset[cols].apply(tuple, axis=1)

        for group_key in valid_groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            mask = group_keys == group_key
            idx = subset.index[mask]

            label = "_".join(str(v) for v in group_key)
            df.loc[idx, "segment_id"] = label
            df.loc[idx, "segment_level"] = level_idx

        still_unassigned = df.loc[remaining_idx, "segment_id"].isna()
        remaining_idx = still_unassigned[still_unassigned].index

    if len(remaining_idx) > 0:
        df.loc[remaining_idx, "segment_id"] = "ALL"
        df.loc[remaining_idx, "segment_level"] = len(hierarchy)

    return df


def segment_sizes(df, segment_col="segment_id"):
    """Convenience helper: row count per assigned segment, sorted descending."""
    return df.groupby(segment_col, observed=True).size().sort_values(ascending=False)

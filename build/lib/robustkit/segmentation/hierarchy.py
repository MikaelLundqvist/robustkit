"""
Hierarchical segmentation with a minimum-size fallback.
"""

import pandas as pd


def hierarchical_segment(df, hierarchy, min_size=20):
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
    return df.groupby(segment_col, observed=True).size().sort_values(ascending=False)

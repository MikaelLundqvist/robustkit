"""
Generic integrity checks, independent of any specific fitting method.
"""


def check_row_integrity(df, group_col):
    total_rows = len(df)
    n_missing_group = int(df[group_col].isna().sum())
    grouped_rows = int(df.groupby(group_col, observed=True).size().sum())

    return {
        "total_rows": total_rows,
        "rows_accounted_for": grouped_rows,
        "rows_missing_group_label": n_missing_group,
        "consistent": total_rows == grouped_rows + 0,
    }


def compare_row_sets(before_df, after_df, key_col):
    before_keys = set(before_df[key_col])
    after_keys = set(after_df[key_col])

    dropped = before_keys - after_keys
    added = after_keys - before_keys

    return {
        "n_before": len(before_keys),
        "n_after": len(after_keys),
        "n_dropped": len(dropped),
        "n_added": len(added),
        "dropped_keys": dropped,
        "added_keys": added,
    }

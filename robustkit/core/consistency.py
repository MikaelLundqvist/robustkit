"""
Generic integrity checks, independent of any specific fitting method.

The goal is to catch silent data loss -- rows that quietly disappear
during filtering, encoding, or grouping without anyone noticing --
before it corrupts a downstream conclusion.
"""


def check_row_integrity(df, group_col):
    """
    Verify that every row in df is assigned to exactly one non-null
    group under group_col, and that grouping accounts for all rows.
    """
    total_rows = len(df)
    n_missing_group = int(df[group_col].isna().sum())
    grouped_rows = int(df.groupby(group_col, observed=True).size().sum())

    return {
        "total_rows": total_rows,
        "rows_accounted_for": grouped_rows,
        "rows_missing_group_label": n_missing_group,
        "consistent": total_rows == grouped_rows + 0,  # grouped excludes NaN groups
    }


def compare_row_sets(before_df, after_df, key_col):
    """
    Compare row identity (by key_col) before and after a filtering or
    fitting step, to make sure no rows were silently dropped or
    duplicated without being explicitly reported.
    """
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

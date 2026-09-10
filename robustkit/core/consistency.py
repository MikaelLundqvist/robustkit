"""
Generic integrity checks, independent of any specific fitting method.
"""

import numpy as np
import pandas as pd


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


def segment_consistency_report(df, segment_col, x_col, y_col, degree=2, min_size=20):
    """
    Run a battery of sanity checks on a segmentation, one row per
    segment -- mirroring checks commonly done by hand before trusting
    a segment's analysis:

      - does the segment meet the recommended minimum size for
        meaningful robust modeling (min_size)?
      - does fitting a Huber trend on this segment use EVERY row in
        it? Robust methods like Huber don't need (or want) outliers
        pre-removed -- they downweight extreme points automatically --
        so any row silently dropped here is due to missing x/y values,
        not intentional filtering, and is worth surfacing rather than
        discovering later as an unexplained discrepancy in row counts.

    Returns one row per segment with n_total, n_valid_xy (rows with
    non-missing x and y), n_dropped_missing_xy, size_ok, fit_ok, and
    fit_error (the exception message, if fitting failed) -- enough to
    see not just THAT a check failed, but why.
    """
    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        n_total = len(group)
        valid = group[[x_col, y_col]].dropna()
        n_valid = len(valid)

        size_ok = n_total >= min_size

        fit_ok = True
        fit_error = None
        if n_valid >= 2:
            try:
                from .trend import fit_huber_trend
                fit_huber_trend(valid[x_col].to_numpy(dtype=float), valid[y_col].to_numpy(dtype=float), degree=degree)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, this IS the check
                fit_ok = False
                fit_error = str(exc)
        else:
            fit_ok = False
            fit_error = "fewer than 2 valid (x, y) pairs"

        rows.append({
            "segment": segment_value,
            "n_total": n_total,
            "n_valid_xy": n_valid,
            "n_dropped_missing_xy": n_total - n_valid,
            "size_ok": size_ok,
            "fit_ok": fit_ok,
            "fit_error": fit_error,
        })

    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)

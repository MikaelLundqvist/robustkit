"""
Run any robustkit.core analysis function independently within each
segment of a dataset, and collect the scalar summary results into a
single report table.

This is deliberately agnostic about which analysis is run: it works
with model_stability_pct, cook_impact, bca_bootstrap_ci, a custom
function you write yourself, or anything else with the signature
analysis_fn(x, y, **kwargs) -> dict.
"""

import numpy as np
import pandas as pd


def _is_reportable_scalar(value):
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return False
    return isinstance(value, (int, float, str, bool, np.integer, np.floating, np.bool_))


def apply_by_segment(df, segment_col, x_col, y_col, analysis_fn, min_points=5, **kwargs):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Typically the output of hierarchical_segment(), but any
        DataFrame with a grouping column works.
    segment_col : str
        Column identifying the segment each row belongs to.
    x_col, y_col : str
        Columns to pass as x and y to analysis_fn for each segment.
    analysis_fn : callable(x, y, **kwargs) -> dict
        Any robustkit.core function (model_stability_pct, cook_impact,
        bca_bootstrap_ci, ...) or a custom function with the same
        signature. Only scalar values in its returned dict are kept
        in the output table -- arrays (e.g. a full fitted curve) are
        intentionally dropped, since this table is meant to be a
        scannable summary rather than a data dump.
    min_points : int
        Segments with fewer rows than this are skipped (marked
        `skipped=True`) rather than passed to analysis_fn, since most
        robust-fitting methods are unreliable or fail outright on very
        small samples.
    **kwargs
        Passed through to analysis_fn on every call.

    Returns
    -------
    pandas.DataFrame
        One row per segment, with `segment`, `n`, `skipped`, and any
        scalar keys returned by analysis_fn. Segments that raised an
        exception get an `error` column instead of failing the whole
        run.
    """
    rows = []

    for segment_value, group in df.groupby(segment_col, observed=True):
        row = {"segment": segment_value, "n": len(group)}

        if len(group) < min_points:
            row["skipped"] = True
            row["reason"] = f"fewer than {min_points} points"
            rows.append(row)
            continue

        row["skipped"] = False

        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)

        try:
            result = analysis_fn(x, y, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: one
            # segment's failure should not abort the whole report
            row["error"] = str(exc)
            rows.append(row)
            continue

        for key, value in result.items():
            if _is_reportable_scalar(value):
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)

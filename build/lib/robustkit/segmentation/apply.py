"""
Run any robustkit.core analysis function independently within each
segment of a dataset, and collect the scalar summary results into a
single report table.
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
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            rows.append(row)
            continue

        for key, value in result.items():
            if _is_reportable_scalar(value):
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)

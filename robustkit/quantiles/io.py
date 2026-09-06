"""
Generic loader for JSON-stat (SCB/PxWeb) statistical tables.

JSON-stat is a standardized, self-describing format for dimensional
statistical data used by Statistics Sweden (SCB) and other national
statistics agencies. Rather than parsing metadata out of column-name
strings (fragile, and specific to one table's phrasing -- the problem
with SCB's wide CSV exports), this loader reads the explicit dimension
structure and reconstructs a tidy long-format table generically, for
any SCB JSON-stat table regardless of how many dimensions it has or
what they're called.

Note on structure: some JSON-stat exports place `id`, `size`, and
`role` as siblings of `dimension` at the dataset level; SCB's exports
(as observed) nest them inside `dimension` instead. This loader reads
them from inside `dimension`, matching SCB's actual export structure.
"""

import itertools
import json

import numpy as np
import pandas as pd


def load_scb_json_stat(path, rename_categories=None):
    """
    Load an SCB JSON-stat export into a tidy long-format DataFrame:
    one row per (dimension combination), with one column per dimension
    (named after its label, e.g. "kön", "år") plus a "value" column.

    Missing values (JSON-stat `null`, and/or positions flagged in the
    optional `status` object -- e.g. SCB's confidentiality
    suppression) are both converted to NaN.

    rename_categories: optional dict of {dimension_label: {old_category: new_category}}
        for merging category labels that changed meaning over time
        (a real SCB quirk: e.g. the oldest-worker age bracket was
        labeled "65-66 år" through 2022 and "65-68 år" from 2023
        onward, following a change in Sweden's public pension age --
        same conceptual bucket, different label depending on year).
        This loader does NOT attempt to detect or merge such cases
        automatically -- that kind of semantic continuity is specific
        to each table and too easy to get silently wrong by guessing.
        Pass an explicit mapping instead, e.g.:
            rename_categories={"ålder": {"65–68 år": "65–66 år"}}

    Returns a DataFrame with columns matching each dimension's label
    (in the table's original dimension order) plus "value".
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    ds = data["dataset"]
    dim_ids = ds["dimension"]["id"]
    values = list(ds["value"])

    # Apply status-flagged positions (e.g. confidentiality suppression)
    # as NaN, in addition to any already-explicit JSON `null` values.
    status = ds.get("status", {})
    for pos_str in status:
        values[int(pos_str)] = None

    dim_labels = {}
    dim_categories = {}
    for dim_id in dim_ids:
        dim = ds["dimension"][dim_id]
        dim_labels[dim_id] = dim["label"]
        cat = dim["category"]
        index_map = cat["index"]
        label_map = cat.get("label", {})
        ordered_codes = sorted(index_map, key=lambda c: index_map[c])
        dim_categories[dim_id] = [label_map.get(c, c) for c in ordered_codes]

    # JSON-stat convention: the flat `value` array has the LAST
    # dimension in `id` varying fastest. itertools.product with lists
    # in `id` order produces exactly that iteration order.
    ordered_label_lists = [dim_categories[d] for d in dim_ids]
    column_names = [dim_labels[d] for d in dim_ids]

    expected_n = 1
    for lst in ordered_label_lists:
        expected_n *= len(lst)
    if expected_n != len(values):
        raise ValueError(
            f"Dimension sizes imply {expected_n} values, but found {len(values)}. "
            "The file may not follow the expected JSON-stat structure."
        )

    rows = []
    for combo, val in zip(itertools.product(*ordered_label_lists), values):
        row = dict(zip(column_names, combo))
        row["value"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df["value"] = df["value"].apply(lambda v: np.nan if v is None else v)

    if rename_categories:
        for col, mapping in rename_categories.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)

    return df

"""
Basic dataset profiling: a quick orientation before running any
information-theoretic analysis.
"""


def profile(df, target=None):
    info = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "missing_per_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }
    if target is not None:
        info["target"] = target
        info["target_dtype"] = str(df[target].dtype)
        info["target_n_unique"] = int(df[target].nunique())
    return info


def print_profile(df, target=None):
    p = profile(df, target=target)
    print(f"Rows: {p['n_rows']}, Columns: {p['n_columns']}")
    if target is not None:
        print(f"Target: {p['target']} (dtype={p['target_dtype']}, unique={p['target_n_unique']})")
    missing = {k: v for k, v in p["missing_per_column"].items() if v > 0}
    if missing:
        print("Missing values:")
        for col, n in missing.items():
            print(f"  {col}: {n}")
    else:
        print("No missing values.")

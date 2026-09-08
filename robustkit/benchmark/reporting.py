"""
Reporting layer on top of segment_position_report: individual-level
deviation reports, running the same benchmark across many grouping
columns at once, and exporting the results to Excel.

Every function here accepts the same benchmark_fit contract as
segment_position_report (a robustkit trend-fit dict, or any object
exposing predict(dataframe)) -- see benchmark.global_model for the
underlying model-agnostic design.
"""

import numpy as np
import pandas as pd

from .global_model import fit_huber_benchmark, benchmark_predict, segment_position_report


def _resolve_benchmark(df, y_col, benchmark_fit, x_col, degree):
    """Shared "fit a default benchmark if none given" logic, used by
    every function in this module so they all fail/behave identically
    when neither benchmark_fit nor x_col is supplied."""
    if benchmark_fit is not None:
        return benchmark_fit
    if x_col is None:
        raise ValueError("x_col must be supplied when no custom benchmark model is provided.")
    return fit_huber_benchmark(df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float), degree=degree)


def residual_summary(df, segment_col, y_col, benchmark_fit=None, x_col=None, degree=2):
    """
    Per-segment residual diagnostics: median residual, MAD (median
    absolute deviation from the segment's own median residual, scaled
    to be comparable to a standard deviation under normality), and the
    10th/90th percentile of residuals.

    Complements segment_position_report: that function summarizes each
    segment with a single (bootstrapped) median difference, which can
    hide whether a segment's residuals are tightly clustered or widely
    spread around that median. residual_summary shows the shape, not
    just the center.
    """
    benchmark_fit = _resolve_benchmark(df, y_col, benchmark_fit, x_col, degree)

    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        y = group[y_col].to_numpy(dtype=float)
        expected = benchmark_predict(benchmark_fit, group, x_col=x_col)
        residuals = y - expected

        median_residual = float(np.median(residuals))
        mad_residual = float(1.4826 * np.median(np.abs(residuals - median_residual)))

        rows.append({
            "segment": segment_value,
            "n": len(group),
            "median_residual": median_residual,
            "mad_residual": mad_residual,
            "p10_residual": float(np.percentile(residuals, 10)),
            "p90_residual": float(np.percentile(residuals, 90)),
        })

    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def negative_deviation_report(df, y_col, benchmark_fit=None, x_col=None, degree=2, top_n=50, id_cols=None):
    """
    Identify the individuals with the largest NEGATIVE deviation from
    a benchmark model -- those furthest below what the benchmark
    predicts, sorted from most negative.

    Intended as underlying material for a conversation (e.g. between
    HR and employee representatives), not as an automatic flag that
    something is wrong: a large negative deviation is a starting point
    for a conversation, not a conclusion on its own.

    id_cols: columns to include so each row can be identified (e.g. an
        employee ID or name column). If omitted, the DataFrame's index
        is included as "row_id" instead -- a bare list of
        actual/expected/difference numbers with no way to identify who
        they belong to is rarely useful, so some identifier is always
        included.
    """
    benchmark_fit = _resolve_benchmark(df, y_col, benchmark_fit, x_col, degree)

    y = df[y_col].to_numpy(dtype=float)
    expected = benchmark_predict(benchmark_fit, df, x_col=x_col)
    difference = y - expected

    if id_cols:
        result = df[id_cols].copy().reset_index(drop=True)
    else:
        result = pd.DataFrame({"row_id": df.index})

    result["actual"] = y
    result["expected"] = expected
    result["difference"] = difference

    return result.sort_values("difference").head(top_n).reset_index(drop=True)


def benchmark_report_suite(df, group_columns, y_col, benchmark_fit=None, x_col=None,
                            degree=2, n_boot=500, ci=95, seed=0):
    """
    Run segment_position_report independently for each column in
    group_columns (e.g. ["Gender", "JobFamily", "Location", ...]),
    reusing the SAME benchmark model across all of them so the results
    are directly comparable to each other -- the benchmark is fit (or
    taken as given) once, not refit separately per grouping column.

    Returns a dict of {column_name: report_dataframe}, one entry per
    entry in group_columns.
    """
    benchmark_fit = _resolve_benchmark(df, y_col, benchmark_fit, x_col, degree)

    return {
        col: segment_position_report(
            df, segment_col=col, y_col=y_col, benchmark_fit=benchmark_fit, x_col=x_col,
            degree=degree, n_boot=n_boot, ci=ci, seed=seed,
        )
        for col in group_columns
    }


def export_benchmark_excel(reports, path):
    """
    Export a dict of {name: DataFrame} (typically the output of
    benchmark_report_suite, but any such dict works) to a single Excel
    workbook, one sheet per entry.

    Sheet names are truncated to Excel's 31-character limit and
    de-duplicated if truncation causes two names to collide.

    Requires openpyxl (an optional dependency: `pip install
    robustkit[excel]`, or `pip install openpyxl` directly). In an
    offline/air-gapped environment where installing an extra package
    isn't possible, use the DataFrames in `reports` directly (e.g.
    write each to CSV instead) rather than this function -- every
    other function in robustkit works without openpyxl installed at
    all.

    Returns `path`, for convenient chaining.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "export_benchmark_excel requires openpyxl, which is not installed. "
            "Install it with `pip install openpyxl` (or `pip install robustkit[excel]`), "
            "or, if that isn't possible in your environment, write each DataFrame in "
            "`reports` to CSV directly instead."
        ) from exc

    seen = {}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, report_df in reports.items():
            sheet_name = str(name)[:31]
            if sheet_name in seen:
                seen[sheet_name] += 1
                suffix = f"_{seen[sheet_name]}"
                sheet_name = sheet_name[: 31 - len(suffix)] + suffix
            else:
                seen[sheet_name] = 0
            report_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return path

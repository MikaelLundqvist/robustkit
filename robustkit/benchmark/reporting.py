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


def deviation_report(df, y_col, benchmark_fit=None, x_col=None, degree=2, top_n=50,
                      direction="negative", id_cols=None):
    """
    Identify the individuals with the largest deviation from a
    benchmark model, sorted by magnitude.

    Intended as underlying material for a conversation (e.g. between
    HR and employee representatives), not as an automatic flag that
    something is wrong: a large deviation is a starting point for a
    conversation, not a conclusion on its own.

    direction: "negative" (default -- furthest BELOW the benchmark,
        sorted most-negative-first; the typical HR/union use case),
        "positive" (furthest ABOVE, sorted most-positive-first), or
        "two_sided" (largest absolute deviation in either direction,
        sorted by |residual| descending).

    id_cols: columns to include so each row can be identified (e.g. an
        employee ID or name column). If omitted, the DataFrame's index
        is included as "row_id" instead -- a bare list of
        actual/expected/residual numbers with no way to identify who
        they belong to is rarely useful, so some identifier is always
        included.

    Returns top_n rows with actual, expected, and residual
    (actual - expected) -- "residual" at this individual level,
    matching mad_outlier_report's convention; contrast with
    segment_position_report's "difference" at the segment level.
    """
    if direction not in ("negative", "positive", "two_sided"):
        raise ValueError(f"direction must be 'negative', 'positive', or 'two_sided', got {direction!r}")

    benchmark_fit = _resolve_benchmark(df, y_col, benchmark_fit, x_col, degree)

    y = df[y_col].to_numpy(dtype=float)
    expected = benchmark_predict(benchmark_fit, df, x_col=x_col)
    residual = y - expected

    if id_cols:
        result = df[id_cols].copy().reset_index(drop=True)
    else:
        result = pd.DataFrame({"row_id": df.index})

    result["actual"] = y
    result["expected"] = expected
    result["residual"] = residual

    if direction == "negative":
        result = result.sort_values("residual")
    elif direction == "positive":
        result = result.sort_values("residual", ascending=False)
    else:
        result = result.reindex(result["residual"].abs().sort_values(ascending=False).index)

    return result.head(top_n).reset_index(drop=True)


def benchmark_report_suite(df, group_columns, y_col, benchmark_fit=None, x_col=None,
                            degree=2, n_boot="auto", ci=95, seed=0):
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
    isn't possible, use export_benchmark_excel_no_deps instead, which
    needs nothing beyond the Python standard library.

    Returns `path`, for convenient chaining.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "export_benchmark_excel requires openpyxl, which is not installed. "
            "Install it with `pip install openpyxl` (or `pip install robustkit[excel]`), "
            "or use export_benchmark_excel_no_deps instead, which needs no extra packages."
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


def export_benchmark_excel_no_deps(reports, path):
    """
    Export a dict of {name: DataFrame} to a minimal but valid .xlsx
    workbook, one sheet per entry, using ONLY the Python standard
    library (zipfile + string templating of the OOXML format) --
    no openpyxl or any other third-party dependency required.

    Use this in offline/air-gapped environments where installing
    openpyxl isn't possible; use export_benchmark_excel instead when
    openpyxl IS available, since it's a more complete, better-tested
    implementation of the Excel format (formatting, formulas, etc. --
    none of which this minimal writer supports).

    Cell values are written as numbers where they parse as one
    (including turning a stray decimal comma into a decimal point),
    and as inline text otherwise. NaN/Inf are written as text, since
    Excel's native format has no representation for them. Sheet names
    are cleaned of Excel-invalid characters and truncated to Excel's
    31-character limit.

    Returns `path`, for convenient chaining.
    """
    import html
    import math
    import re
    import zipfile

    def _clean_xml_text(value):
        if value is None:
            return ""
        text = str(value)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)  # invalid XML control chars
        return html.escape(text, quote=True)

    def _clean_sheet_name(name):
        invalid = ["\\", "/", "*", "?", ":", "[", "]"]
        for ch in invalid:
            name = name.replace(ch, "")
        return name.strip()[:31]

    def _col_letter(n):
        result = ""
        while n:
            n, r = divmod(n - 1, 26)
            result = chr(65 + r) + result
        return result

    def _write_cell(cell_ref, val):
        try:
            num = float(val.replace(",", ".")) if isinstance(val, str) else float(val)
            if math.isnan(num) or math.isinf(num):
                raise ValueError
            return f'<c r="{cell_ref}"><v>{num}</v></c>'
        except (TypeError, ValueError):
            text = _clean_xml_text(val)
            return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    seen = {}
    sheet_names = []
    for name in reports:
        candidate = _clean_sheet_name(str(name))
        if candidate in seen:
            seen[candidate] += 1
            suffix = f"_{seen[candidate]}"
            candidate = candidate[: 31 - len(suffix)] + suffix
        else:
            seen[candidate] = 0
        sheet_names.append(candidate)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        overrides = "".join(
            f'    <Override PartName="/xl/worksheets/sheet{i}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n'
            for i in range(1, len(reports) + 1)
        )
        z.writestr("[Content_Types].xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{overrides}</Types>
""")

        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""")

        sheet_entries = "".join(
            f'        <sheet name="{_clean_xml_text(sn)}" sheetId="{i}" r:id="rId{i}"/>\n'
            for i, sn in enumerate(sheet_names, start=1)
        )
        z.writestr("xl/workbook.xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
{sheet_entries}    </sheets>
</workbook>
""")

        rels = "".join(
            f'    <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>\n'
            for i in range(1, len(reports) + 1)
        )
        z.writestr("xl/_rels/workbook.xml.rels", f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}</Relationships>
""")

        for i, (_, df) in enumerate(reports.items(), start=1):
            rows_xml = f'<row r="1">' + "".join(
                f'<c r="{_col_letter(c)}1" t="inlineStr"><is><t>{_clean_xml_text(col)}</t></is></c>'
                for c, col in enumerate(df.columns, start=1)
            ) + "</row>"

            for r_i, (_, row) in enumerate(df.iterrows(), start=2):
                cells = "".join(
                    _write_cell(f"{_col_letter(c_i)}{r_i}", val)
                    for c_i, val in enumerate(row, start=1)
                )
                rows_xml += f'<row r="{r_i}">{cells}</row>'

            dim = f"A1:{_col_letter(max(df.shape[1], 1))}{df.shape[0] + 1}"
            z.writestr(f"xl/worksheets/sheet{i}.xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <dimension ref="{dim}"/>
    <sheetData>{rows_xml}</sheetData>
</worksheet>
""")

    return path

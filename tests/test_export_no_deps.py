import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from robustkit import export_benchmark_excel, export_benchmark_excel_no_deps


def make_test_reports():
    df1 = pd.DataFrame({
        "segment": ["A", "B", "C"], "n": [30, 45, 12],
        "difference": [123.456, -78.9, 0.0], "note": ["ok", "flagged: <5%>", "special & chars"],
    })
    df2 = pd.DataFrame({"x": [1, 2], "y": [float("nan"), 3.14]})
    return {
        "gender": df1,
        "very_long_sheet_name_that_exceeds_31_chars_A": df2,
        "very_long_sheet_name_that_exceeds_31_chars_B": df1,
    }


def test_export_no_deps_creates_valid_file():
    reports = make_test_reports()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.xlsx")
        result = export_benchmark_excel_no_deps(reports, path)
        assert result == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100


def test_export_no_deps_readable_by_openpyxl():
    pytest.importorskip("openpyxl")
    import openpyxl

    reports = make_test_reports()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.xlsx")
        export_benchmark_excel_no_deps(reports, path)

        wb = openpyxl.load_workbook(path)
        assert len(wb.sheetnames) == 3


def test_export_no_deps_matches_openpyxl_sheet_names_and_dedup():
    pytest.importorskip("openpyxl")
    import openpyxl

    reports = make_test_reports()
    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "a.xlsx")
        path_b = os.path.join(tmp, "b.xlsx")
        export_benchmark_excel(reports, path_a)
        export_benchmark_excel_no_deps(reports, path_b)

        wb_a = openpyxl.load_workbook(path_a)
        wb_b = openpyxl.load_workbook(path_b)
        assert wb_a.sheetnames == wb_b.sheetnames


def test_export_no_deps_matches_openpyxl_numeric_values():
    pytest.importorskip("openpyxl")
    import openpyxl

    reports = {"data": pd.DataFrame({"a": [1, 2, 3], "b": [1.5, -2.25, 0.0]})}
    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "a.xlsx")
        path_b = os.path.join(tmp, "b.xlsx")
        export_benchmark_excel(reports, path_a)
        export_benchmark_excel_no_deps(reports, path_b)

        ws_a = openpyxl.load_workbook(path_a)["data"]
        ws_b = openpyxl.load_workbook(path_b)["data"]

        rows_a = list(ws_a.iter_rows(values_only=True))
        rows_b = list(ws_b.iter_rows(values_only=True))
        assert len(rows_a) == len(rows_b)
        for r_a, r_b in zip(rows_a[1:], rows_b[1:]):  # skip header row (strings, compared elsewhere)
            for v_a, v_b in zip(r_a, r_b):
                assert float(v_a) == float(v_b)


def test_export_no_deps_handles_special_xml_characters():
    reports = {"data": pd.DataFrame({"note": ["<tag>", "a & b", 'quote"here']})}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.xlsx")
        export_benchmark_excel_no_deps(reports, path)

        pytest.importorskip("openpyxl")
        import openpyxl
        ws = openpyxl.load_workbook(path)["data"]
        values = [row[0] for row in ws.iter_rows(min_row=2, values_only=True)]
        assert values == ["<tag>", "a & b", 'quote"here']


def test_export_no_deps_requires_no_external_packages(monkeypatch):
    """
    Simulate an environment without openpyxl (or any other Excel
    library) and confirm export_benchmark_excel_no_deps still works --
    the whole point of this function.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("openpyxl", "xlsxwriter") or name.startswith("openpyxl.") or name.startswith("xlsxwriter."):
            raise ImportError(f"simulated: no module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reports = make_test_reports()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.xlsx")
        result = export_benchmark_excel_no_deps(reports, path)
        assert os.path.exists(result)

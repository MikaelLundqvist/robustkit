"""
Validation for robustkit.quantiles.io -- generic JSON-stat loading.

Each check names the specific claim (about load_scb_json_stat's
behavior) it verifies, in the spirit of requirements-based testing:
know WHAT you're checking before you check it, not just "run it and
see if it crashes."
"""

import json
import tempfile
from pathlib import Path

from robustkit import load_scb_json_stat

from ..openml_suite.common import section, safe_run

DATA_DIR = Path(__file__).parent / "data"


def _make_minimal_fixture(tmp_path, include_status=False):
    """
    A minimal, self-contained JSON-stat fixture matching SCB's real
    export structure (id/size/role nested inside "dimension", not as
    siblings) -- 2 x 2 x 3 = 12 values.
    """
    dataset = {
        "dimension": {
            "Kon": {"label": "kön", "category": {"index": {"1": 0, "2": 1}, "label": {"1": "män", "2": "kvinnor"}}},
            "Grp": {"label": "grupp", "category": {"index": {"A": 0, "B": 1}, "label": {"A": "Grupp A", "B": "Grupp B"}}},
            "Tid": {
                "label": "år",
                "category": {
                    "index": {"2020": 0, "2021": 1, "2022": 2},
                    "label": {"2020": "2020", "2021": "2021", "2022": "2022"},
                },
            },
            "id": ["Kon", "Grp", "Tid"],
            "size": [2, 2, 3],
            "role": {"time": ["Tid"]},
        },
        "value": list(range(100, 112)),
    }
    if include_status:
        dataset["value"][5] = None
        dataset["status"] = {"5": ".."}

    path = tmp_path / "fixture.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset}, f)
    return path


def run_io_edge_cases():
    section("robustkit.quantiles.io -- edge cases (no real data required)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # CLAIM: dimensions are reconstructed correctly from a flat
        # value array (JSON-stat convention: last `id` dimension
        # varies fastest).
        path = _make_minimal_fixture(tmp_path)

        def check_shape():
            df = load_scb_json_stat(path)
            assert df.shape == (12, 4), f"expected (12,4), got {df.shape}"
            assert set(df.columns) == {"kön", "grupp", "år", "value"}
            return df

        df = safe_run("dimension reconstruction: shape and column names", check_shape)
        if df is not None:
            print(f"         shape={df.shape}, columns={list(df.columns)}")

        # CLAIM: both JSON `null` values and `status`-flagged positions
        # become NaN.
        path2 = _make_minimal_fixture(tmp_path, include_status=True)

        def check_missing():
            df2 = load_scb_json_stat(path2)
            n_missing = int(df2["value"].isna().sum())
            assert n_missing == 1, f"expected 1 missing value, got {n_missing}"
            return n_missing

        n_missing = safe_run("null + status positions both convert to NaN", check_missing)
        if n_missing is not None:
            print(f"         missing values: {n_missing} (expected 1)")

        # CLAIM: rename_categories merges the given categories.
        def check_rename():
            df3 = load_scb_json_stat(path, rename_categories={"grupp": {"Grupp B": "Grupp A"}})
            assert set(df3["grupp"]) == {"Grupp A"}
            assert (df3["grupp"] == "Grupp A").sum() == 12
            return df3

        result = safe_run("rename_categories merges two categories into one", check_rename)
        if result is not None:
            print(f"         all 12 rows merged into 'Grupp A'")

        # CLAIM: a dimension-size mismatch raises ValueError rather
        # than silently truncating or misaligning the data.
        with open(path, encoding="utf-8") as f:
            bad_data = json.load(f)
        bad_data["dataset"]["value"] = bad_data["dataset"]["value"][:-1]
        bad_path = tmp_path / "bad_fixture.json"
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)

        def check_size_mismatch_raises():
            try:
                load_scb_json_stat(bad_path)
            except ValueError:
                return True
            raise AssertionError("Expected ValueError on dimension-size mismatch, none raised")

        safe_run("dimension-size mismatch raises ValueError (not silent corruption)", check_size_mismatch_raises)


def run_io_against_real_files():
    section("robustkit.quantiles.io -- against real SCB exports")

    quartiles_path = DATA_DIR / "quartiles.json"
    age_path = DATA_DIR / "age.json"

    if not quartiles_path.exists() or not age_path.exists():
        print(f"  (skipped -- place real SCB JSON-stat exports at "
              f"{quartiles_path} and {age_path} to run this check)")
        return

    def check_quartiles():
        df1 = load_scb_json_stat(quartiles_path)
        assert df1.shape == (360, 5), f"expected (360,5), got {df1.shape}"
        assert df1["value"].isna().sum() == 0
        return df1

    df1 = safe_run("real quartile table: shape and completeness", check_quartiles)
    if df1 is not None:
        print(f"         shape={df1.shape}, 0 missing (as expected for this table)")

    def check_age():
        df2 = load_scb_json_stat(age_path)
        assert df2.shape == (432, 5), f"expected (432,5), got {df2.shape}"
        n_missing = int(df2["value"].isna().sum())
        assert n_missing == 37, f"expected 37 missing, got {n_missing}"
        return df2

    df2 = safe_run("real age table: shape and known missing-value count", check_age)
    if df2 is not None:
        print(f"         shape={df2.shape}, {df2['value'].isna().sum()} missing "
              f"(as expected: confidentiality-suppressed cells)")

        def check_pension_age_merge():
            df2_renamed = load_scb_json_stat(age_path, rename_categories={"ålder": {"65–68 år": "65–66 år"}})
            merged_count = int((df2_renamed["ålder"] == "65–66 år").sum())
            original_6566 = int((df2["ålder"] == "65–66 år").sum())
            original_6568 = int((df2["ålder"] == "65–68 år").sum())
            assert merged_count == original_6566 + original_6568
            return merged_count, original_6566, original_6568

        result = safe_run(
            "rename_categories merges '65-68 år' into '65-66 år' (2023 pension-age reform)",
            check_pension_age_merge,
        )
        if result is not None:
            merged, a, b = result
            print(f"         merged count: {merged} = {a} + {b}")


def main():
    run_io_edge_cases()
    run_io_against_real_files()


if __name__ == "__main__":
    main()

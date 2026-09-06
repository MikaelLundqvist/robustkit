import json

import numpy as np
import pandas as pd
import pytest

from robustkit import load_scb_json_stat, plot_quantile_trend, quantile_trend_dispersion
from robustkit.quantiles.trend import prepare_quantile_trend


def make_json_stat_fixture(tmp_path, include_status=False):
    """
    A minimal, self-contained JSON-stat fixture matching SCB's actual
    export structure (id/size/role nested inside "dimension", not as
    siblings) -- 2 categories x 2 categories x 3 years = 12 values.
    """
    dataset = {
        "dimension": {
            "Kon": {
                "label": "kön",
                "category": {"index": {"1": 0, "2": 1}, "label": {"1": "män", "2": "kvinnor"}},
            },
            "Grp": {
                "label": "grupp",
                "category": {"index": {"A": 0, "B": 1}, "label": {"A": "Grupp A", "B": "Grupp B"}},
            },
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


def test_load_scb_json_stat_shape_and_columns(tmp_path):
    path = make_json_stat_fixture(tmp_path)
    df = load_scb_json_stat(path)

    assert df.shape == (12, 4)  # 2*2*3 rows, 3 dimension columns + value
    assert set(df.columns) == {"kön", "grupp", "år", "value"}
    assert df["value"].isna().sum() == 0


def test_load_scb_json_stat_handles_null_and_status(tmp_path):
    path = make_json_stat_fixture(tmp_path, include_status=True)
    df = load_scb_json_stat(path)

    assert df["value"].isna().sum() == 1


def test_load_scb_json_stat_rename_categories(tmp_path):
    path = make_json_stat_fixture(tmp_path)
    df = load_scb_json_stat(path, rename_categories={"grupp": {"Grupp B": "Grupp A"}})

    assert set(df["grupp"]) == {"Grupp A"}
    assert (df["grupp"] == "Grupp A").sum() == 12  # all rows merged into one category


def test_load_scb_json_stat_raises_on_size_mismatch(tmp_path):
    path = make_json_stat_fixture(tmp_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["dataset"]["value"] = data["dataset"]["value"][:-1]  # remove one value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ValueError):
        load_scb_json_stat(path)


def make_quantile_table():
    return pd.DataFrame({
        "year": [2020, 2021, 2022, 2023],
        "lower": [28000, 29000, 30500, 32000],
        "med": [35000, 36500, 38000, 40000],
        "upper": [44000, 45500, 47500, 50000],
    })


def test_prepare_quantile_trend_sorts_and_renames():
    df = make_quantile_table().sample(frac=1, random_state=0)  # shuffle rows
    trend = prepare_quantile_trend(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")

    assert list(trend.columns) == ["year", "q1", "median", "q3"]
    assert (trend["year"].diff().dropna() > 0).all()  # sorted ascending
    assert (trend["q3"] >= trend["median"]).all()
    assert (trend["median"] >= trend["q1"]).all()


def test_plot_quantile_trend_returns_prepared_data():
    import matplotlib
    matplotlib.use("Agg")

    df = make_quantile_table()
    trend = plot_quantile_trend(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")

    assert list(trend["year"]) == [2020, 2021, 2022, 2023]


def test_quantile_trend_dispersion_computes_ratio():
    df = make_quantile_table()
    result = quantile_trend_dispersion(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")

    expected_2020 = (44000 - 28000) / 35000
    assert result.loc[result["year"] == 2020, "dispersion_ratio"].iloc[0] == pytest.approx(expected_2020)


def test_quantile_trend_dispersion_zero_median_returns_zero():
    df = pd.DataFrame({"year": [2020], "lower": [-5], "med": [0], "upper": [5]})
    result = quantile_trend_dispersion(df, x_col="year", q1_col="lower", median_col="med", q3_col="upper")
    assert result["dispersion_ratio"].iloc[0] == 0.0

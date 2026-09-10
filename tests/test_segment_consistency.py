import numpy as np
import pandas as pd

from robustkit import segment_consistency_report


def make_test_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    job_family = np.array(rng.choice(["ENG", "FIN"], n, p=[0.7, 0.3]))
    job_family[:15] = "ADM"  # deliberately tiny, below default min_size=20
    age = rng.uniform(25, 60, n).astype(float)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)

    df = pd.DataFrame({"JobFamily": job_family, "age": age, "salary": salary})

    eng_idx = df[df["JobFamily"] == "ENG"].index[:10]
    df.loc[eng_idx, "age"] = np.nan  # deliberately inject missing values

    return df


def test_segment_consistency_report_structure():
    df = make_test_df()
    report = segment_consistency_report(df, segment_col="JobFamily", x_col="age", y_col="salary", min_size=20)

    expected_cols = {"segment", "n_total", "n_valid_xy", "n_dropped_missing_xy", "size_ok", "fit_ok", "fit_error"}
    assert set(report.columns) == expected_cols
    assert set(report["segment"]) == {"ADM", "ENG", "FIN"}


def test_segment_consistency_report_detects_missing_values():
    df = make_test_df()
    report = segment_consistency_report(df, segment_col="JobFamily", x_col="age", y_col="salary", min_size=20)

    eng_row = report[report["segment"] == "ENG"].iloc[0]
    assert eng_row["n_dropped_missing_xy"] == 10
    assert eng_row["n_valid_xy"] == eng_row["n_total"] - 10

    fin_row = report[report["segment"] == "FIN"].iloc[0]
    assert fin_row["n_dropped_missing_xy"] == 0


def test_segment_consistency_report_flags_undersized_segment():
    df = make_test_df()
    report = segment_consistency_report(df, segment_col="JobFamily", x_col="age", y_col="salary", min_size=20)

    adm_row = report[report["segment"] == "ADM"].iloc[0]
    assert adm_row["n_total"] == 15
    assert adm_row["size_ok"] is False or adm_row["size_ok"] == False  # noqa: E712

    eng_row = report[report["segment"] == "ENG"].iloc[0]
    assert eng_row["size_ok"] == True  # noqa: E712


def test_segment_consistency_report_flags_unfittable_segment():
    df = make_test_df()
    tiny_df = pd.DataFrame({"JobFamily": ["X"], "age": [30.0], "salary": [40000.0]})
    combined = pd.concat([df, tiny_df], ignore_index=True)

    report = segment_consistency_report(combined, segment_col="JobFamily", x_col="age", y_col="salary", min_size=20)
    x_row = report[report["segment"] == "X"].iloc[0]

    assert x_row["fit_ok"] == False  # noqa: E712
    assert "fewer than 2" in x_row["fit_error"]


def test_segment_consistency_report_fit_ok_for_well_formed_segments():
    df = make_test_df()
    report = segment_consistency_report(df, segment_col="JobFamily", x_col="age", y_col="salary", min_size=20)
    assert report["fit_ok"].all()
    assert report["fit_error"].isna().all()

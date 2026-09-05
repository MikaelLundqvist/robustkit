import pandas as pd

from robustkit import classify_quadrants


def test_classify_quadrants_basic():
    df = pd.DataFrame({"metric_a": [1, 2, 3, 4], "metric_b": [10, 20, 5, 15]})
    classified = classify_quadrants(df, x_col="metric_a", y_col="metric_b")

    assert "quadrant" in classified.columns
    assert set(classified["quadrant"]) <= {"high_high", "high_x_only", "high_y_only", "low_low"}
    assert classified.attrs["x_col"] == "metric_a"
    assert classified.attrs["y_col"] == "metric_b"


def test_classify_quadrants_custom_labels():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 5, 15]})
    classified = classify_quadrants(
        df, x_col="a", y_col="b",
        labels={"high_high": "star", "low_low": "weak"},
    )
    # Unspecified keys fall back to generic defaults
    assert set(classified["quadrant"]) <= {"star", "high_x_only", "high_y_only", "weak"}


def test_classify_quadrants_thresholds_stored_in_attrs():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    classified = classify_quadrants(df, x_col="a", y_col="b")
    assert classified.attrs["x_threshold"] == df["a"].median()
    assert classified.attrs["y_threshold"] == df["b"].median()

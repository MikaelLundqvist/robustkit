"""
Dataset configuration and loading for the openml_suite validation
scripts.

Each entry in DATASETS names an OpenML dataset plus which columns to
use as the continuous x, the target y, an optional categorical
segment, and a small set of feature columns for the information/
benchmark modules. If OpenML's column names don't match what's
configured here (versions and mirrors occasionally differ), the
calling script prints the actual available columns so this file can
be corrected.
"""

from sklearn.datasets import fetch_openml


DATASETS = [
    {
        "name": "house_prices (Ames Housing)",
        "openml_name": "house_prices",
        "openml_version": 1,
        "x_col": "GrLivArea",
        "y_col": "SalePrice",
        "segment_col": "Neighborhood",
        "feature_cols": ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"],
    },
    {
        "name": "abalone",
        "openml_name": "abalone",
        "openml_version": 1,
        "x_col": "Shell_weight",
        "y_col": "Class_number_of_rings",
        "segment_col": "Sex",
        "feature_cols": ["Length", "Diameter", "Height", "Whole_weight", "Shell_weight"],
    },
    {
        "name": "wine-quality-red",
        "openml_name": "wine-quality-red",
        "openml_version": 1,
        "x_col": "alcohol",
        "y_col": "class",
        "segment_col": None,
        "feature_cols": ["alcohol", "volatile_acidity", "sulphates", "citric_acid", "pH"],
    },
    {
        "name": "autoMpg",
        "openml_name": "autoMpg",
        "openml_version": 1,
        "x_col": "horsepower",
        "y_col": "class",
        "segment_col": "origin",
        "feature_cols": ["horsepower", "weight", "displacement", "acceleration"],
    },
    {
        # Included specifically to independently cross-check earlier
        # exploratory findings (via Copilot) on this same dataset:
        # RM/ZN expected "robust"; TAX/CHAS expected "structural
        # sensitivity"; AGE/CRIM expected "fragile".
        "name": "boston (housing)",
        "openml_name": "boston",
        "openml_version": 1,
        "x_col": "LSTAT",
        "y_col": "MEDV",
        "segment_col": None,
        "feature_cols": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
    },
    {
        # Large n (~54000), strongly right-skewed price -- a good
        # stress test for scale and for the report module's
        # analyst/publisher-view distinction.
        "name": "diamonds",
        "openml_name": "diamonds",
        "openml_version": 1,
        "x_col": "carat",
        "y_col": "price",
        "segment_col": "cut",
        "feature_cols": ["carat", "depth", "table"],
    },
]


def load_dataset(cfg):
    data = fetch_openml(name=cfg["openml_name"], version=cfg["openml_version"], as_frame=True, parser="auto")
    df = data.frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

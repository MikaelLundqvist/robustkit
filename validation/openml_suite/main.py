"""
Run the full openml_suite validation: every robustkit module against
every configured dataset in datasets.py.

This is NOT a pass/fail test suite. Results are meant to be read and
interpreted: some "failures" (a function raising on a genuinely
unsuitable dataset/column combination) are informative findings, not
bugs to fix blindly. Every step is wrapped (see common.safe_run) so
one failure doesn't abort the whole run.

Requires internet access on first run (fetch_openml downloads and then
locally caches each dataset). Run with:

    python -m validation.openml_suite.main

To run validation against just one dataset or one robustkit module
during development, import the relevant pieces directly instead --
see this package's __init__.py docstring for an example.
"""

import warnings

from .common import section
from .datasets import DATASETS, load_dataset
from .run_core import run_core_module
from .run_segmentation import run_segmentation_module
from .run_information import run_information_module
from .run_benchmark import run_benchmark_module
from .run_report import run_report_module

warnings.filterwarnings("ignore")


def run_all(datasets=None):
    """Run the full suite. Pass a subset of DATASETS to run only some."""
    import matplotlib
    matplotlib.use("Agg")

    for cfg in (datasets or DATASETS):
        section(f"DATASET: {cfg['name']}")

        try:
            df = load_dataset(cfg)
            print(f"  [OK]   fetch_openml({cfg['openml_name']!r})")
        except Exception as exc:
            print(f"  [FAIL] fetch_openml({cfg['openml_name']!r}): {type(exc).__name__}: {exc}")
            continue

        print(f"  Shape: {df.shape}")
        preview_cols = list(df.columns)[:15]
        print(f"  Columns (first 15): {preview_cols}{' ...' if df.shape[1] > 15 else ''}")

        missing_cols = [c for c in (cfg["x_col"], cfg["y_col"]) if c not in df.columns]
        if missing_cols:
            print(f"  [FAIL] Expected column(s) not found: {missing_cols}")
            print(f"  All available columns: {list(df.columns)}")
            continue

        x, y = run_core_module(df, cfg["x_col"], cfg["y_col"])
        run_segmentation_module(df, cfg["x_col"], cfg["y_col"], cfg["segment_col"])
        run_information_module(df, cfg["y_col"], cfg["feature_cols"])
        run_benchmark_module(df, cfg["x_col"], cfg["y_col"], cfg["segment_col"], cfg["feature_cols"])
        run_report_module(x, y)

    section("VALIDATION RUN COMPLETE")


if __name__ == "__main__":
    run_all()

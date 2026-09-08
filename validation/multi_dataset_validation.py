"""
Backward-compatible entry point.

The validation suite previously lived entirely in this one file; it
has since been split into validation/openml_suite/ (one module per
robustkit module: run_core.py, run_segmentation.py, run_information.py,
run_benchmark.py, run_report.py, plus datasets.py and common.py) for
easier standalone editing and review.

This file just forwards to the new location so existing habits
(`python validation/multi_dataset_validation.py`) keep working.
Prefer running `python -m validation.openml_suite.main` directly, or
importing individual pieces from validation.openml_suite for
standalone work on one dataset or one robustkit module.
"""

from validation.openml_suite.main import run_all

if __name__ == "__main__":
    run_all()

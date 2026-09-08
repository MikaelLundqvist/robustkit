"""
openml_suite -- validation of robustkit against real, well-known
datasets fetched via sklearn.datasets.fetch_openml.

Split into one module per robustkit module (run_core, run_segmentation,
run_information, run_benchmark, run_report) plus shared helpers
(common.py) and dataset configuration (datasets.py), so each part can
be read, run, or extended independently -- e.g. to add a dataset, edit
only datasets.py; to change how robustkit.core is exercised, edit only
run_core.py.

Run the full suite with:

    python -m validation.openml_suite.main

or import and run individual pieces, e.g.:

    from validation.openml_suite.datasets import DATASETS, load_dataset
    from validation.openml_suite.run_core import run_core_module

    df = load_dataset(DATASETS[0])
    run_core_module(df, DATASETS[0]["x_col"], DATASETS[0]["y_col"])
"""

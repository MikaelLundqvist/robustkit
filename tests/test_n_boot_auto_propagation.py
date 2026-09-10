import numpy as np
import pandas as pd

from robustkit import (
    bca_bootstrap_ci, bca_bootstrap_ci_by_index, segment_position_report,
    benchmark_report_suite, segment_benchmark_report, segment_benchmark_drilldown_report,
)
from robustkit.core.uncertainty import _resolve_n_boot


def test_bca_bootstrap_ci_defaults_to_auto():
    import inspect
    sig = inspect.signature(bca_bootstrap_ci)
    assert sig.parameters["n_boot"].default == "auto"


def test_bca_bootstrap_ci_by_index_defaults_to_auto():
    import inspect
    sig = inspect.signature(bca_bootstrap_ci_by_index)
    assert sig.parameters["n_boot"].default == "auto"


def test_bca_bootstrap_ci_auto_produces_valid_interval():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 200)
    y = 1000 + 50 * x + rng.normal(0, 300, 200)
    result = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_))
    assert result["lower"] <= result["estimate"] <= result["upper"]


def make_mixed_size_df(seed=0):
    rng = np.random.default_rng(seed)
    n_big, n_small = 25000, 50
    department = np.array(["Big"] * n_big + ["Small"] * n_small)
    age = rng.uniform(25, 60, n_big + n_small)
    salary = 30000 + 400 * age + rng.normal(0, 800, n_big + n_small)
    return pd.DataFrame({"department": department, "age": age, "salary": salary})


def test_segment_position_report_n_boot_defaults_to_auto():
    import inspect
    sig = inspect.signature(segment_position_report)
    assert sig.parameters["n_boot"].default == "auto"


def test_segment_position_report_auto_resolves_per_segment():
    """A large segment should internally use fewer bootstrap
    iterations than a small one, resolved independently per segment."""
    df = make_mixed_size_df()
    report = segment_position_report(df, segment_col="department", x_col="age", y_col="salary")

    assert (report["ci_available"] == True).all()  # noqa: E712
    big_row = report[report["segment"] == "Big"].iloc[0]
    small_row = report[report["segment"] == "Small"].iloc[0]
    assert big_row["ci_lower"] <= big_row["ci_upper"]
    assert small_row["ci_lower"] <= small_row["ci_upper"]


def test_benchmark_report_suite_n_boot_defaults_to_auto():
    import inspect
    sig = inspect.signature(benchmark_report_suite)
    assert sig.parameters["n_boot"].default == "auto"


def test_segment_benchmark_report_n_boot_defaults_to_auto():
    import inspect
    sig = inspect.signature(segment_benchmark_report)
    assert sig.parameters["n_boot"].default == "auto"


def test_segment_benchmark_drilldown_report_n_boot_defaults_to_auto():
    import inspect
    sig = inspect.signature(segment_benchmark_drilldown_report)
    assert sig.parameters["n_boot"].default == "auto"


def test_explicit_n_boot_still_respected_everywhere():
    """Regression: explicit n_boot must still override 'auto' exactly,
    across every function that gained the new default."""
    df = make_mixed_size_df(seed=1)

    report = segment_position_report(df, segment_col="department", x_col="age", y_col="salary", n_boot=17)
    assert (report["ci_available"] == True).all()  # noqa: E712

    result = bca_bootstrap_ci_by_index(300, lambda idx: 0.0, n_boot=17)
    assert result["lower"] == result["upper"] == 0.0  # degenerate but must not crash with n_boot=17

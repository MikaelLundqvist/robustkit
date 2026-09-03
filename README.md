# robustkit

> ⚠️ **Under active development.** This is an early placeholder release
> to claim the package name on PyPI. The API is incomplete and may
> change without notice. Not yet recommended for production use.

Practical tools for robust analysis of a single continuous relationship:
y as a function of one continuous x.

The guiding idea: **a conclusion that survives multiple fitting methods
is more trustworthy than one that only holds under a single model.**
`robustkit` makes it easy to compare Huber, Tukey biweight, and OLS
fits side by side, identify and quantify the influence of individual
observations, and get honest, bias-corrected uncertainty estimates.

## Status

`robustkit.core` (trend fitting, stability, diagnostics, uncertainty,
consistency checks) and `robustkit.segmentation` (hierarchical
grouping, per-segment analysis) are stable and tested.
Information-theoretic feature selection is under active development
and not yet included in this version.

## Installation

```bash
git clone https://github.com/<your-username>/robustkit.git
cd robustkit
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from robustkit import (
    fit_huber_trend, fit_tukey_trend, predict_trend,
    model_stability_pct, cooks_diagnostic, cook_impact,
    bootstrap_band, bca_bootstrap_ci,
)

# x: a single continuous predictor, y: a single continuous outcome
x = np.random.default_rng(0).uniform(20, 60, 200)
y = 1000 + 50 * x - 0.4 * x**2 + np.random.default_rng(1).normal(0, 500, 200)

# Fit and predict
fit = fit_huber_trend(x, y, degree=2)
y_pred = predict_trend(fit, x_new=[30, 40, 50])

# Does the conclusion survive a change of fitting method?
stability = model_stability_pct(x, y)
print("Median % spread between Huber/Tukey/OLS:", stability["median_pct_diff"])

# Which points are influential, and how much do they matter?
diag = cooks_diagnostic(x, y)
impact = cook_impact(x, y, diag["flagged_indices"])
print("Median % change in curve if flagged points removed:", impact["median_pct_change"])

# Uncertainty
band = bootstrap_band(x, y)
ci = bca_bootstrap_ci(x, y, statistic_fn=lambda x_, y_: np.median(y_))
```

See `examples/quickstart_tutorial.py` for a complete, runnable walkthrough.

## Segmentation

Run any `robustkit.core` analysis independently across subgroups of a
larger dataset, with automatic fallback to coarser groupings when a
finer one is too small to analyze reliably:

```python
from robustkit import hierarchical_segment, apply_by_segment, model_stability_pct

# hierarchy: finest to coarsest grouping
hierarchy = [["department", "level", "status"], ["level", "status"], ["status"]]
segmented = hierarchical_segment(df, hierarchy, min_size=20)

report = apply_by_segment(
    segmented, segment_col="segment_id", x_col="age", y_col="value",
    analysis_fn=model_stability_pct,
)
```

`apply_by_segment` works with any function shaped like
`analysis_fn(x, y, **kwargs) -> dict` -- built-in ones
(`model_stability_pct`, `cook_impact`, `bca_bootstrap_ci`, ...) or your
own. Only scalar values in the returned dict end up in the report
table; segments below `min_points` are skipped rather than causing an
error.

## Design principles

- **One continuous x, one continuous y** at the core. This keeps every
  function's output visually and numerically interpretable (a curve
  you can plot, a band you can read).
- **Diagnosis and action are separate steps.** `cooks_diagnostic`
  flags candidates; `cook_impact` tells you whether removing them
  actually changes anything.
- **OLS is a reference point, not the enemy.** Comparing robust fits
  against OLS is how you know whether robustness mattered at all.

## License

MIT -- see [LICENSE](LICENSE).

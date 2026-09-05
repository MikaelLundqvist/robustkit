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
consistency checks), `robustkit.segmentation` (hierarchical grouping,
per-segment analysis), and `robustkit.information` (mutual-information
feature ranking, quadrant classification, and pairwise
redundancy/synergy scoring) are stable and tested.

**Note on `information_efficiency`:** values can exceed 1.0 for
continuous features. `mutual_information` is estimated on the
full-resolution continuous values, while `entropy_bits` is computed on
a binned version of the same feature (since `entropy()` expects
categorical input). Binning discards information, so `entropy_bits` is
a lower bound on the feature's true entropy -- an efficiency above 1.0
signals that the feature carries more usable information than a coarse
categorical summary of it would capture. This is expected behavior,
not a bug.

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

fit = fit_huber_trend(x, y, degree=2)
y_pred = predict_trend(fit, x_new=[30, 40, 50])

stability = model_stability_pct(x, y)
print("Median % spread between Huber/Tukey/OLS:", stability["median_pct_diff"])

diag = cooks_diagnostic(x, y)
impact = cook_impact(x, y, diag["flagged_indices"])
print("Median % change in curve if flagged points removed:", impact["median_pct_change"])

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

## Feature ranking (information)

Rank features by mutual information with a target, normalized by each
feature's own entropy, and classify them into four quadrants:

```python
from robustkit import rank_features, quadrant_report, plot_feature_space

ranking = rank_features(df, target="value")
report = quadrant_report(df, target="value")   # adds a `quadrant` column
plot_feature_space(df, target="value")          # same quadrants, visualized
```

`quadrant_report` and `plot_feature_space` always agree on quadrant
assignment -- both route through the same thresholding logic.

**Caveat:** default thresholds are the *median* mutual information /
efficiency across the ranked features. With only a handful of
features, this can put a genuinely weak feature in the same "high"
half as a strong one, since roughly half of any list sits above its
own median regardless of how large the actual gap is. Median
thresholding becomes meaningful with a reasonably large feature set;
for a handful of candidates, read the raw `mutual_information` /
`information_efficiency` values directly rather than relying on the
quadrant label alone.

See `examples/information_tutorial.py` for a complete walkthrough.

## Feature pairing (information)

Beyond ranking single features, evaluate *pairs* of features together:
how redundant are they with each other, and does knowing one reveal
additional predictive value in the other (synergy, e.g. an interaction
effect)?

```python
from robustkit import (
    conditional_mutual_information, communication_score,
    rank_by_communication, pair_redundancy, pair_synergy,
    rank_communicative_pairs,
)

# How communicable is a single feature -- not just predictive, but
# suitable for a clear chart/table (adequate group sizes, homogeneous
# groups, few enough categories to show at once)?
comm_ranking = rank_by_communication(df, target="value")

# How much does region's relevance to the target change once
# department is already known?
synergy = pair_synergy(df, feature_1="department", feature_2="region", target="value")

# Rank every candidate pair by combined relevance, penalizing
# redundant pairs and rewarding genuine synergy
pairs = rank_communicative_pairs(df, target="value")
```

All mutual-information-based quantities in this module (`rank_features`,
`conditional_mutual_information`, `pair_redundancy`, `pair_synergy`,
`communication_score`) are expressed in **bits**, consistent with
`entropy()` -- internally, scikit-learn's MI estimators return nats
and are converted before being used anywhere in this package.

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

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
per-segment analysis), `robustkit.information` (mutual-information
feature ranking, quadrant classification, pairwise redundancy/synergy
scoring), `robustkit.benchmark` (global-trend segment comparison,
Robustness Map), `robustkit.report` (analyst vs. publisher views,
dispersion measures), and `robustkit.quantiles` (generic JSON-stat
loading, published-quantile-trend visualization, and lognormal-
calibrated reconstruction of individual-level data from aggregated
summaries) are stable and tested.

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

## Trend growth rate and goodness of fit

```python
from robustkit import trend_derivative, goodness_of_fit, compare_polynomial_degrees

fit = fit_huber_trend(df["age"], df["salary"])

# Rate of change of the trend itself (e.g. "salary growth per year of
# age"), not just its level
rates = trend_derivative(fit, x=[30, 40, 50])

# How well does this fit actually explain the variation in y?
goodness_of_fit(df["age"], df["salary"], degree=2)

# Don't assume a quadratic trend is always the right choice -- check
# empirically whether a higher degree captures meaningfully more
compare_polynomial_degrees(df["age"], df["salary"], degrees=(1, 2, 3, 4))
```

**Note:** x is standardized internally before building polynomial
features (both here and throughout `robustkit.core`), since raw
polynomial features become numerically unstable at higher degrees for
realistic x scales (e.g. age^5 vastly outscales age^1). This is
transparent to callers -- `predict_trend` and `trend_derivative` still
take and return values in the original x scale.

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

## Benchmarking against a global trend

Compare each segment's observed outcome against what a single global
robust trend predicts, with bootstrap uncertainty on the difference --
answers "which groups deviate from the overall trend, and by how
much?" rather than "how does the trend look overall?":

```python
from robustkit import segment_position_report

report = segment_position_report(
    df, segment_col="department", x_col="age", y_col="salary",
)
#   segment    n  observed_median  expected_median  difference  ci_lower  ci_upper
#   Finance  176        48339.70         47799.82      539.88    202.15   1031.01
#        HR  174        45718.84         46647.39     -928.55  -1293.26   -580.36
#        IT  250        47226.50         47126.91       99.59   -117.56    510.81
```

A segment's confidence interval crossing zero means no clear deviation
from the benchmark; HR and Finance above don't cross zero, IT does.

## Robustness Map

Classify features by how much a conclusion about their relationship
with the target depends on (a) fitting method choice and (b) specific
influential observations -- two genuinely different failure modes that
a single diagnostic can miss:

```python
from robustkit import feature_robustness_report, plot_feature_robustness

report = feature_robustness_report(df, target="value")
#   feature  stability_pct  cook_impact_pct    quadrant
#      CRIM          8.9             17.1     fragile
#       AGE         16.8             15.5     fragile
#        RM          4.9              0.1     robust
#       TAX         22.2              1.4     structural_sensitivity

plot_feature_robustness(report=report)
```

Four quadrants: **robust** (low spread, low impact), **structural
sensitivity** (sensitive to fitting method, not to specific points),
**data sensitive** (a few points drive the conclusion, method choice
barely matters), **fragile** (both -- least trustworthy).

`quadrant_report`/`plot_feature_space` (information) and
`feature_robustness_report`/`plot_feature_robustness` (benchmark) both
route through the same shared classifier, `robustkit.classify_quadrants`
-- any future quadrant-based analysis in this package will too.

## Analyst view vs. publisher view

Two visualizations that look superficially similar but answer
genuinely different questions:

```python
from robustkit import plot_analyst_view, plot_publisher_view, dispersion_ratio, iqr

# "How confident are we in the trend estimate?" -- a bootstrap
# confidence band that SHRINKS as sample size grows.
plot_analyst_view(df["age"], df["salary"])

# "How spread out are actual values in the population?" -- a median +
# IQR band that does NOT shrink with more data, since it reflects
# real dispersion, not estimation uncertainty. show_points defaults to
# False, since this view is meant for publishing potentially sensitive
# data (e.g. individual salaries) without exposing raw points.
plot_publisher_view(df["age"], df["salary"])
```

This distinction matters in practice: with 20x more data (same
underlying distribution), the analyst view's confidence band roughly
halves in width, while the publisher view's IQR band stays essentially
unchanged -- confirmed by the package's own test suite.

`dispersion_ratio(y)` -- (Q3-Q1)/median -- and `iqr(y)` are available
standalone for tabular reporting; `dispersion_by_bin(x, y, n_bins=10)`
computes both across bins of a continuous x, e.g. to check whether
dispersion (inequality) grows with age.

## Loading published quantile tables (SCB / JSON-stat)

Some statistics agencies (e.g. Statistics Sweden, SCB) publish
quantiles (Q1/median/Q3) directly, with no individual-level data
available at all. `robustkit.quantiles` loads these tables generically
via JSON-stat, a standardized dimensional-data format used by SCB and
other national statistics agencies -- avoiding the fragility of
parsing metadata out of column-name strings in a wide CSV export.

```python
from robustkit import load_scb_json_stat, plot_quantile_trend, quantile_trend_dispersion

df = load_scb_json_stat("some_scb_table.json")

# A real SCB quirk this loader does NOT try to guess automatically:
# category labels can change meaning over time (e.g. Sweden's oldest
# working-age bracket was labeled "65-66 år" through 2022 and
# "65-68 år" from 2023, following a pension-age reform). Merge such
# cases explicitly:
df = load_scb_json_stat(
    "some_scb_table.json",
    rename_categories={"ålder": {"65–68 år": "65–66 år"}},
)

# Once reshaped to a wide table with q1/median/q3 columns:
plot_quantile_trend(wide_df, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
quantile_trend_dispersion(wide_df, x_col="år", q1_col="q1", median_col="median", q3_col="q3")
```

This is the "quantiles are already given" case. A complementary case
-- reconstructing approximate individual-level data from aggregated
group means, for when only summary statistics (not quantiles) are
available -- is planned as a follow-up (`robustkit.quantiles.reconstruct`).

See `examples/quantiles_tutorial.py` for a complete walkthrough.

## Reconstructing individual-level data from aggregated summaries

For the complementary case -- only aggregated group summaries (n,
Q1, median, Q3) are available, not the quantile trend itself as the
final answer, and you want to run `robustkit.core` analyses as if
individual data existed:

```python
from robustkit import expand_aggregated_table, check_reconstruction_quality, fit_huber_trend

# One row per group (e.g. year), with n/q1/median/q3 columns
synthetic = expand_aggregated_table(
    summary_df, n_col="n", q1_col="q1", median_col="median", q3_col="q3",
    group_cols=["year"], value_name="salary",
)

# Now usable exactly like real individual-level data:
fit = fit_huber_trend(synthetic["year"], synthetic["salary"])
```

Method: a lognormal distribution is calibrated (via the IQR) to match
each group's reported Q1/median/Q3, then `n` synthetic values are
drawn from it. Validated end-to-end against real published SCB salary
data: a Huber trend fitted on reconstructed pseudo-individual data
tracked the true published median trend within 2% across 12 years.

**Note on what this recovers:** because a Huber (or Tukey) fit on
right-skewed reconstructed data tracks something close to the
*median* trend it was calibrated against -- not the arithmetic mean --
this is consistent with, not a limitation of, the reconstruction
method. To target the mean instead, fit on `log(value)` and
exponentiate predictions back, which approximates the geometric mean.

Always check `check_reconstruction_quality()` before trusting a
reconstruction: real Q1/median/Q3 triples aren't always perfectly
consistent with a pure lognormal shape.

**Warning -- unbounded tail at large n:** a lognormal has no natural
upper limit, and its expected maximum grows with n. Reconstructing at
the TRUE group size from a national table (SCB salary tables can
report n in the hundreds of thousands to millions) can produce
implausibly extreme tail values -- real salaries have practical
ceilings a pure lognormal doesn't know about. This package's own
examples and tests deliberately scale n down to a few thousand for
demonstration; calibration quality (matching Q1/median/Q3) doesn't
depend on reproducing the true population size, but tail plausibility
does. No clipping is applied automatically.

### When only a mean is available (no quantiles at all)

Some tables (e.g. SCB's age-breakdown salary tables) report only a
mean per group, with no spread information. Two deliberately separate
methods are provided, each making a different explicit assumption --
compare them rather than silently picking one:

```python
from robustkit import (
    expand_aggregated_table_flat, expand_aggregated_table_borrowed_dispersion,
    compare_reconstruction_methods,
)

# Method 1: repeat the mean n times -- zero within-group spread.
# Recovers between-group regression coefficients reasonably well
# (validated in the original technique this is based on) but
# understates individual-level variation.
flat = expand_aggregated_table_flat(df, n_col="n", mean_col="mean_salary", group_cols=["age"])

# Method 2: borrow a dispersion_ratio from a DIFFERENT table that does
# report quantiles, and use it to imply an approximate spread around
# the mean. Stacks two assumptions (mean-as-median, and that the
# borrowed ratio transfers to this population) -- illustrative, not a
# substitute for genuine quantile data for this specific table.
borrowed = expand_aggregated_table_borrowed_dispersion(
    df, n_col="n", mean_col="mean_salary", dispersion_ratio=0.45, group_cols=["age"],
)

# Compare both for a single group directly:
compare_reconstruction_methods(n=2000, mean=52200, dispersion_ratio=0.45)
```

Both methods are documented with their specific assumptions rather
than presented as equally valid defaults -- being explicit about which
assumption was made lets the analyst judge how much a conclusion
depends on it, rather than presenting an assumption as a measurement.

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

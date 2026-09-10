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
model-agnostic custom benchmarks, residual/deviation reporting, Excel
export, Robustness Map), `robustkit.report` (analyst vs. publisher
views, dispersion measures, combined Huber+IQR view),
`robustkit.quantiles` (generic JSON-stat loading, published-quantile-
trend visualization, and lognormal-calibrated reconstruction of
individual-level data from aggregated summaries), and
`robustkit.segment_awareness` (automatic hierarchical segmentation +
analysis, no manual hierarchy construction required) are stable and
tested.

**Recent fixes from real-dataset validation:**
- `rank_features`/`quadrant_report`/`rank_communicative_pairs` no
  longer crash on pandas `Categorical` columns containing missing
  values (found via OpenML's Boston Housing dataset).
- `bootstrap_band` (and `plot_analyst_view`, which uses it) now
  defaults to `n_boot="auto"`, scaling iterations down for large
  datasets since each iteration refits a full Huber model -- found to
  become impractically slow at n_boot=200 on a ~54,000-row dataset.
  Pass an explicit integer to opt out and always use exactly that many
  iterations.

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

**Sanity-checking a segmentation before trusting it:**
`segment_consistency_report` runs a small battery of checks per
segment -- does it meet the recommended minimum size, and does fitting
a Huber trend on it use every row (robust methods don't need outliers
pre-removed, so a silently dropped row usually means a missing x/y
value slipped through, not intentional filtering):

```python
from robustkit import segment_consistency_report

segment_consistency_report(df, segment_col="department", x_col="age", y_col="salary", min_size=20)
#   segment  n_total  n_valid_xy  n_dropped_missing_xy  size_ok  fit_ok  fit_error
```

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

Compare each segment's observed outcome against what a benchmark model
predicts, with bootstrap uncertainty on the difference -- answers
"which groups deviate from the overall trend, and by how much?" rather
than "how does the trend look overall?":

```python
from robustkit import segment_position_report

# Default: a single global Huber trend on one continuous x
report = segment_position_report(
    df, segment_col="department", x_col="age", y_col="salary",
)
#   segment    n  observed_median  expected_median  difference  ci_lower  ci_upper  ci_available
#   Finance  176        48339.70         47799.82      539.88    202.15   1031.01          True
#        HR  174        45718.84         46647.39     -928.55  -1293.26   -580.36          True
#        IT  250        47226.50         47126.91       99.59   -117.56    510.81          True
```

A segment's confidence interval crossing zero means no clear deviation
from the benchmark; HR and Finance above don't cross zero, IT does.

**Custom, model-agnostic benchmarks:** the default single-column Huber
trend can be replaced with any richer model -- e.g. one using age,
age-squared, job level, overtime status, and a reference cluster
together, rather than a single x. Provide any object exposing
`predict(dataframe) -> array`:

```python
report = segment_position_report(
    df, segment_col="department", y_col="salary", benchmark_fit=my_richer_model,
)
```

`segment_position_report` never inspects what the model uses
internally -- it only calls `predict()`.

**Small segments:** groups with fewer than `MIN_POINTS_FOR_CI` (default
20) observations still get `observed_median` / `expected_median` /
`difference`, but `ci_lower` / `ci_upper` are `NaN` and
`ci_available` is `False` -- a BCa bootstrap confidence interval (which
relies on a jackknife step) is not attempted for populations that
small, since it can fail outright or become statistically meaningless.
For segments at or above the threshold, the interval is a full BCa
(bias-corrected and accelerated) bootstrap interval, via the same
`bca_bootstrap_ci_by_index` primitive used elsewhere in the package --
not a plain percentile bootstrap.

## Reporting: residuals, individual deviations, batch runs, and Excel export

Four functions built on the same benchmark contract as
`segment_position_report`, for turning a benchmark into something a
non-technical audience (or a spreadsheet) can use directly:

```python
from robustkit import (
    residual_summary, negative_deviation_report,
    benchmark_report_suite, export_benchmark_excel,
)

# Per-segment residual SHAPE (not just the median difference):
residual_summary(df, segment_col="job_family", y_col="salary", x_col="age")
#   segment    n  median_residual  mad_residual  p10_residual  p90_residual

# Individuals furthest BELOW the benchmark, sorted most-negative-first --
# material for a conversation, not an automatic flag:
negative_deviation_report(
    df, y_col="salary", x_col="age", top_n=50, id_cols=["employee_id"],
)
#   employee_id     actual   expected  difference

# Run the same benchmark across several grouping columns at once,
# reusing ONE fitted benchmark so results are directly comparable:
reports = benchmark_report_suite(
    df, group_columns=["gender", "job_family", "location"], y_col="salary", x_col="age",
)
# -> {"gender": DataFrame, "job_family": DataFrame, "location": DataFrame}

# Every report as its own sheet in one workbook:
export_benchmark_excel(reports, "salary_report.xlsx")
```

All four accept the same `benchmark_fit` / `x_col` contract as
`segment_position_report` (default single-column Huber trend, or any
custom model exposing `predict(dataframe)`).

**Dependency-free Excel export:** `export_benchmark_excel` requires
`openpyxl` (an optional dependency). In an offline/air-gapped
environment where installing it isn't possible, use
`export_benchmark_excel_no_deps` instead -- identical interface,
implemented with only the Python standard library (writes valid
`.xlsx` files via `zipfile` and OOXML templating directly, no
third-party package required):

```python
from robustkit import export_benchmark_excel_no_deps

export_benchmark_excel_no_deps(reports, "salary_report.xlsx")
```

Prefer `export_benchmark_excel` when `openpyxl` is available -- it's a
more complete, better-tested implementation of the Excel format.

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

## Segment awareness: automatic hierarchical grouping + analysis

`segment_stability_report` and `segment_benchmark_report` build the
hierarchical segmentation automatically from a flat, most-specific-
first list of columns, then run an existing analysis within the
result -- no separate `hierarchical_segment(...)` + `apply_by_segment(...)`
preparation step required:

```python
from robustkit import segment_stability_report, segment_benchmark_report

# hierarchy built automatically: [JobFamily, Level, OT] -> [Level, OT] -> [OT] -> ALL
report = segment_stability_report(
    df, x_col="age", y_col="salary",
    segment_cols=["JobFamily", "Level", "OT"], min_size=20,
)

report = segment_benchmark_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"],
    x_col="age", min_size=20,  # or benchmark_fit=my_custom_model
)
```

Both add a `segment_level` column showing which tier of the hierarchy
each reported segment actually landed on (0 = finest), so a fallback
to a coarser grouping is visible rather than silent. These are pure
convenience wrappers -- identical results to building the hierarchy
by hand with `hierarchical_segment` and calling `apply_by_segment` /
`segment_position_report` directly.

`mad_outlier_report` flags individuals whose residual is an outlier
relative to their OWN segment's typical spread (MAD), not the whole
population -- built on the same automatic hierarchical segmentation as
above, so even someone in a small segment is compared against a
sensibly-sized reference group rather than an irrelevant one:

```python
from robustkit import mad_outlier_report

report = mad_outlier_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age",
    min_size=20, k=3.0, direction="negative", id_cols=["employee_id"],
)
#   employee_id     actual   expected   residual  residual_pct    segment  segment_level  segment_mad  threshold  flagged
```

`direction`: `"negative"` (default -- flag underperformance relative
to the benchmark), `"positive"`, or `"two_sided"`. Flagging compares
each residual to `k` MADs from its *own segment's* median residual
(not literally zero), so a segment the benchmark is systematically
biased for doesn't get every member flagged just for that bias.
Segments with zero MAD (a degenerate case, usually a tiny segment
where every residual happens to match) are treated as having an
infinite threshold rather than flagging everyone in them.

`export_outlier_pdf` renders one chart per segment -- built from the
same segmentation and flagging as `mad_outlier_report` -- as a
one-page-per-segment PDF, for visual verification alongside the
numeric report:

```python
from robustkit import export_outlier_pdf

export_outlier_pdf(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age",
    path="outliers.pdf", min_size=20, k=3.0, id_cols=["employee_id"],
)
```

Each page plots every observation in that segment, the benchmark's
expected values (the exact same values used for flagging, not a
separately re-fit curve), and flagged outliers marked distinctly.
Segments with fewer than `min_points_to_plot` (default 5) observations
are skipped in the PDF -- a chart with a handful of points isn't
meaningfully verifiable -- but still appear in `mad_outlier_report`'s
numeric output. The idea: a numeric flag and a visual confirmation are
two independent checks, and agreement between them is stronger
evidence than either alone.

### Drilldown reports: every hierarchy level at once, without exclusive assignment

`segment_stability_report`, `segment_benchmark_report`, and
`mad_outlier_report` each assign every individual to exactly ONE
segment (their most specific grouping meeting `min_size`). That
answers "what is the single most relevant reference population for
THIS individual?"

`segment_benchmark_drilldown_report` and `mad_outlier_drilldown_report`
answer a different question -- "what does every granularity level look
like on its own?" -- by reporting EVERY level of the hierarchy
independently, without exclusive assignment. The same individual can
appear in multiple rows (e.g. once in a `JobFamily x Level x OT` row,
and again in the broader `Level x OT` row), whenever both groupings
independently meet `min_size`:

```python
from robustkit import segment_benchmark_drilldown_report, mad_outlier_drilldown_report

segment_benchmark_drilldown_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age", min_size=20,
)
mad_outlier_drilldown_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age", k=3.0,
)
```

Both add a `segment_level` column, and the sum of `n` across rows will
exceed the population size -- that's the expected signature of
overlap, not a bug. Use the exclusive functions when you need to route
each individual to one home; use the drilldown functions when you want
to see every level side by side.

## Combined model + spread view

`plot_analyst_view` and `plot_publisher_view` each show one thing --
estimation uncertainty, or population spread -- deliberately kept
separate. `plot_huber_iqr` shows both together: one or more trend
curves overlaid with median + IQR error bars, plus an optional
residual-quality box, matching the combined model-and-spread diagram
style common in salary/wage analysis reporting:

```python
from robustkit import plot_huber_iqr

result = plot_huber_iqr(df["age"], df["salary"], degree=2, bins=15)
# result["grid"], result["huber"], result["binned"]
```

`show_points` defaults to `False`, consistent with `plot_publisher_view`.

**Multiple curves, bootstrap bands, and full style control:**

```python
plot_huber_iqr(
    df["age"], df["salary"],
    methods=("huber", "tukey", "ols", "median_ensemble"),  # overlay all four
    show_bootstrap_band=True, bootstrap_levels=(95, 50),    # nested confidence bands
    cap_style="manual",         # hand-drawn boxplot-style Q1/Q3 "hats" instead of matplotlib's default caps
    residual_box_metric="mdape",  # MdAPE + IQR(resid) instead of R^2/MAE/RMSE
    ylim="dynamic",              # y-limits set from the data (min*0.95, max*1.05)
    style={
        "huber_line": {"color": "red", "linewidth": 2, "linestyle": "-", "label": "Huber poly(2)"},
        "iqr_color": "black", "cap_width": 0.15,
    },
)
```

`methods` selects which trend curve(s) to draw (`"tukey"` and
`"median_ensemble"` -- the pointwise median of Huber/Tukey/OLS --
require statsmodels). `style` overrides individual colors, line
widths, and other visual details without needing to touch anything
else; every new parameter here defaults to the original, simpler
single-Huber-curve appearance, so existing calls are unaffected.

**Two grouping strategies:** `grouping="bin"` (default) uses quantile-
based binning for stable estimates even in small populations.
`grouping="unique"` instead groups by each EXACT x value (e.g. every
individual age in years) -- matching a workbook-style `groupby(x)`
aggregation -- and, when a given x value has fewer than
`min_n_for_iqr` (default 5) observations, omits its IQR error bar
entirely rather than showing an unreliable one:

**Caveat, found via validation against a real dataset:** `grouping="unique"`
only makes sense for x values with natural repetition (e.g. integer
ages) -- for a genuinely continuous, high-precision measurement (e.g.
carat weight to several decimal places), nearly every x value is
unique, so almost nothing meets `min_n_for_iqr` and the result shows
no IQR bars at all. Use `grouping="bin"` (the default) for
high-precision continuous x; reserve `grouping="unique"` for x values
that naturally repeat.

```python
plot_huber_iqr(
    df["age"], df["salary"], grouping="unique", min_n_for_iqr=5,
)
```

The absence of an error bar at a given age is itself information --
it signals the sample at that exact value is too small to say
anything about spread, not just a plotting simplification.

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

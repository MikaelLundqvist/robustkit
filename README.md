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
not a plain percentile bootstrap. That primitive's jackknife step
(needed for the acceleration term) is inherently O(n) regardless of
`n_boot`, which becomes the dominant cost on large real segments --
verified directly on a ~185,000-row segment (minutes, unusable) --
so above `jackknife_cap` observations (1000 by default), a random
subsample of that many positions is left out one at a time instead of
every single one. Confidence interval bounds from the capped and
uncapped versions were verified to differ by roughly 0.000005 on that
same real segment, while cutting its runtime to about 3 seconds; pass
`jackknife_cap=None` to any function built on
`bca_bootstrap_ci_by_index` (including `segment_contribution_report`
and `segment_benchmark_drilldown_report`) to always use the full,
uncapped jackknife instead.

## Reporting: residuals, individual deviations, batch runs, and Excel export

Four functions built on the same benchmark contract as
`segment_position_report`, for turning a benchmark into something a
non-technical audience (or a spreadsheet) can use directly:

```python
from robustkit import (
    residual_summary, deviation_report,
    benchmark_report_suite, export_benchmark_excel,
)

# Per-segment residual SHAPE (not just the median difference):
residual_summary(df, segment_col="job_family", y_col="salary", x_col="age")
#   segment    n  median_residual  mad_residual  p10_residual  p90_residual

# Individuals furthest from the benchmark, sorted by residual --
# material for a conversation, not an automatic flag:
deviation_report(
    df, y_col="salary", x_col="age", top_n=50, id_cols=["employee_id"],
)
#   employee_id     actual   expected  residual

# direction="negative" (default, furthest below), "positive" (furthest
# above), or "two_sided" (largest |residual| either direction).

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

Both accept an optional `title=None` to override the default title
(e.g. `plot_analyst_view(x, y, title="Q3 salary review")`), as does
`plot_quantile_trend`.

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

`segment_quality_report` combines several independent robustness
measures into a single row per segment -- model stability, Cook's
impact, MdAPE, and IQR of residuals -- so one table shows how
trustworthy each segment's conclusion is, rather than cross-
referencing several separate function calls by hand:

```python
from robustkit import segment_quality_report

segment_quality_report(
    df, x_col="age", y_col="salary", segment_cols=["JobFamily", "Level", "OT"], min_size=20,
)
#   segment  n  skipped  median_pct_diff  max_pct_diff  n_flagged  cook_impact_pct  mdape  iqr_resid  segment_level
```

Each measure is computed and error-handled independently -- if
model stability fails (e.g. Tukey needs statsmodels, which may not be
installed), Cook's impact, MdAPE, and IQR are still reported for that
segment rather than the whole row failing.

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

`reference`: `"model"` (default) flags against `benchmark_fit` (or, if
none is given, a single Huber trend fit once across the whole
population). `"curve"` instead fits a fresh Huber trend WITHIN each
segment, on that segment's own data alone, and flags against THAT --
no benchmark model involved at all. Use `"curve"` when the flagging
itself, not just a chart, should match "how does this person compare
to their immediate segment-mates" -- e.g. a report for someone who
shouldn't need the underlying benchmark model explained to trust the
numbers. `"benchmark_curve"` is a third option in between: like
`"model"`, it starts from `benchmark_fit`'s raw predictions, but then,
WITHIN each segment, fits a Huber trend to those predictions (not to
the actual y values, as `"curve"` does) and flags against that smoothed
curve -- the exact reference `export_outlier_pdf`'s `mode="benchmark"`
draws, so pairing the two gives full consistency without discarding
the benchmark model's other covariates the way `"curve"` does.
`x_col` is required for all three, but `"curve"` and `"benchmark_curve"`
need it even when a custom `benchmark_fit` would otherwise have made it
optional. `mad_outlier_drilldown_report` takes the same `reference`
parameter, with identical semantics.

`export_outlier_pdf` renders one chart per segment -- built from the
same segmentation and flagging as `mad_outlier_report` -- as a
one-page-per-segment PDF, for visual verification alongside the
numeric report:

```python
from robustkit import export_outlier_pdf

export_outlier_pdf(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age",
    path="outliers.pdf", min_size=20, k=3.0, id_cols=["employee_id"],
    annotate_flagged_with="employee_id",  # optional: label flagged points directly
)
```

`mode="benchmark"` (default) plots a Huber curve fit to the benchmark's
own predicted values, not the raw values connected point-to-point --
for a multivariate benchmark model (depending on more than `x_col`),
raw expected values aren't a pure function of `x_col` within a
segment, so connecting them can zig-zag sharply even for a perfectly
sensible model. The Huber-fit curve smooths over that without becoming
a different reference: it's still derived entirely from the
benchmark's own predictions. `mode="local"` instead fits a fresh Huber
trend within each segment on the actual observed values alone -- often
easier to read for "where does this person sit relative to their
immediate colleagues?"

`mode` (what's drawn) and `reference` (what flagging actually uses)
are independent choices -- `export_outlier_pdf` takes the same
`reference="model"/"curve"/"benchmark_curve"` parameter as
`mad_outlier_report`, and always computes flagging exactly the way
that report would with matching arguments, so the chart and the
numeric report never disagree. Pairing `mode="local"` with
`reference="curve"`, or `mode="benchmark"` with
`reference="benchmark_curve"`, gives a chart where the drawn curve and
the flagging rule are the *same thing* -- no benchmark model to
explain at all in the first case, or the full benchmark model's
covariates preserved (just smoothed for display) in the second --
useful for charts handed to someone who should just be able to look
and trust it. Other combinations -- e.g. `mode="benchmark"` with
`reference="curve"` -- are allowed too (draw the benchmark's smoothed
curve for context, but flag against each segment's own local trend);
the info box always states plainly which curve is drawn and which
reference the flagging rule actually used, so no combination is
ambiguous on the page itself.

`segment_mode="exclusive"` (default) vs. `segment_mode="drilldown"`
offers the same choice as `export_huber_iqr_pdf`: one page per
individual's single most-specific segment, or a page for every
qualifying group at every hierarchy level independently (with the
expected overlap -- the same individual can appear on multiple pages).

Each page plots every observation in that segment, the reference
curve, flagged outliers marked distinctly, and an info box (n, flagged
count/percentage, the segment's MAD, the flagging rule, and which
curve/reference is in play) so a reviewer can see *why* points are
flagged directly from the chart, without cross-referencing the numeric
report.
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
from robustkit import (
    segment_benchmark_drilldown_report, mad_outlier_drilldown_report,
    segment_stability_drilldown_report, outlier_drilldown_summary,
)

segment_benchmark_drilldown_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age", min_size=20,
)
mad_outlier_drilldown_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age", k=3.0,
)
segment_stability_drilldown_report(
    df, x_col="age", y_col="salary", segment_cols=["JobFamily", "Level", "OT"], min_size=20,
)
```

Both add a `segment_level` column, and the sum of `n` across rows will
exceed the population size -- that's the expected signature of
overlap, not a bug. Use the exclusive functions when you need to route
each individual to one home; use the drilldown functions when you want
to see every level side by side.

**A conclusion that survives multiple reference populations is more
trustworthy than one that only holds under a single, narrow
definition of "expected".** `outlier_drilldown_summary` counts how
many hierarchy levels each individual was flagged on:

```python
drilldown = mad_outlier_drilldown_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age",
    k=3.0, id_cols=["employee_id"],
)
outlier_drilldown_summary(drilldown, id_col="employee_id")
#   employee_id  outlier_levels          levels
```

An individual flagged at every level (finest grouping AND every
broader fallback) is a stronger candidate than one that only appears
as an outlier under one specific, narrow segmentation -- a Cook's-
distance-style robustness check, applied to the choice of reference
population rather than to individual data points.

### Reference Population Sensitivity: a fourth robustness axis

`robustkit` already checks robustness against fitting method
(Huber/Tukey/OLS, `core`), individual observations (Cook's impact,
`core.diagnostics`), and segment granularity (the drilldown functions
above). `dual_reference_outlier_report` adds a fourth axis: robustness
against WHICH reference population an individual is compared to.

```python
from robustkit import dual_reference_outlier_report

report = dual_reference_outlier_report(
    df, y_col="salary", x_col="age", segment_cols=["JobFamily", "Level", "OT"],
    min_size=20, k=3.0, id_cols=["employee_id"],
)
#   employee_id  ...  local_flagged  global_flagged  reference_type
```

Each individual's residual is computed against two different
references simultaneously:
- **local** -- their own segment's Huber trend, fit fresh on just that
  segment. Answers "how does this person deviate from their immediate
  colleagues?"
- **global** -- the benchmark model (default single-column Huber, or
  a custom multi-column model). Answers "how does this person deviate
  from the organization's expected structure?"

`reference_type` classifies each individual: **"A"** (outlier under
both -- the strongest candidates, low regardless of how "expected" is
defined), **"B"** (outlier locally only -- often a tightly-clustered
segment where a modest dip stands out among close peers but not
against the wider population), **"C"** (outlier globally only --
often a whole segment trending low while each individual is typical
within it), or **"D"** (neither -- the normal case).

**A subtlety that matters for what "global" actually means:** the
global reference's MAD/median is computed across the ENTIRE
population, not re-centered per segment. If it were centered per
segment, a segment uniformly shifted below the benchmark would have
that shift silently absorbed by the per-segment median subtraction --
nobody in it would ever register as a global outlier, no matter how
far the whole segment sits from the benchmark, since everyone would
share roughly the same "typical" residual for their segment. This was
found by testing, not designed in from the start: an earlier version
reused `mad_outlier_report`'s per-segment MAD for the "global" side
and it could never produce a genuine Type C, because per-segment
recentering makes "global" behave like a second "local" view. It's
also worth knowing that if segment-level shifts are large relative to
individual noise AND affect a large share of the population, the
population-wide MAD naturally inflates to accommodate that variation
-- so a single shifted segment competing against several
similarly-large ones may not clear the threshold either; Type C shows
up most clearly when one segment's shift is a real minority pattern
against an otherwise-homogeneous majority, not when most segments
already differ substantially from each other.

`dual_reference_outlier_drilldown_report` extends the same A/B/C/D
classification to every hierarchy level at once, without exclusive
assignment -- the drilldown counterpart, same overlap semantics as
the other drilldown functions:

```python
from robustkit import dual_reference_outlier_drilldown_report

dual_reference_outlier_drilldown_report(
    df, y_col="salary", x_col="age", segment_cols=["JobFamily", "Level", "OT"], k=3.0,
)
#   ...  segment  segment_level  reference_type
```

The GLOBAL reference is still computed once across the whole
population (not per level) for the same reason as above; only the
LOCAL reference is refit fresh within each group at each level.

### Where does the effect come from: segment_contribution_report

`segment_benchmark_report`/`-drilldown_report` answer "how far from
benchmark is this segment?" `segment_contribution_report` answers a
different question: **where in the hierarchy does that gap actually
arise?**

```python
from robustkit import segment_contribution_report

segment_contribution_report(
    df, y_col="salary", segment_cols=["JobFamily", "Level", "OT"], x_col="age", min_size=20,
)
#   segment  segment_level  ...  difference  parent_segment  contribution
```

Each segment's `contribution` is `child_difference - parent_difference`
-- the part of its benchmark gap NOT already explained by its broader,
coarser parent. A JobFamily within a Level+OT combination might show a
large `difference` from benchmark simply because that whole Level+OT
combination runs high or low (a STRUCTURAL effect, already visible one
level up); `contribution` isolates whatever remains once that broader
pattern is subtracted out -- the LOCAL effect specific to this finer
grouping. Segments at the coarsest hierarchy level have no parent
within `segment_cols`, so `contribution` is `NaN` and `parent_segment`
is `None` for them.

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

**Per-segment PDF export:** `export_huber_iqr_pdf` renders one
`plot_huber_iqr` chart per segment -- built from the same automatic
hierarchical segmentation as the rest of `segment_awareness` -- as a
one-page-per-segment PDF, suitable for sharing with the people each
chart describes:

```python
from robustkit import export_huber_iqr_pdf

export_huber_iqr_pdf(
    df, x_col="age", y_col="salary", segment_cols=["JobFamily", "Level", "OT"],
    path="segments.pdf", min_size=20,
)
```

Domain-specific segment naming (e.g. translating an internal code like
`"ENG_P3_Yes"` into a readable label such as "Engineers, level 3,
overtime-eligible") stays entirely outside the package via an optional
`title_fn(segment_name, info) -> str` callback, where `info` is
`{"n", "segment_level"}`:

```python
def pretty_title(segment_code, info):
    return f"{my_own_translation(segment_code)} (n={info['n']})"

export_huber_iqr_pdf(..., title_fn=pretty_title)
```

`export_outlier_pdf` accepts the same `title_fn(segment_name, info)`
callback for symmetry, with `info` there being `{"n", "n_flagged",
"pct_flagged", "k", "segment_mad", "segment_level"}` -- useful if
outlier review PDFs also need domain-specific titles, even though for
many teams that PDF stays internal and doesn't need one. Both default
to a generic title when `title_fn` is omitted, so existing calls are
unaffected.

**`mode="exclusive"` (default) vs. `mode="drilldown"`**, matching the
same distinction as the report functions above: `"exclusive"` assigns
each individual to exactly one, most-specific segment -- one page per
final segment, no overlap. `"drilldown"` instead renders a page for
EVERY qualifying group at EVERY hierarchy level independently -- e.g.
a page for `"ENG_P3_Yes"` AND a separate page for the broader
`"P3_Yes"`, which includes those same individuals alongside everyone
else at that level. The same individual can appear on multiple pages
in drilldown mode -- that's the expected signature of overlap, same as
the drilldown reports:

```python
export_huber_iqr_pdf(..., mode="drilldown")
```

All of `plot_huber_iqr`'s style parameters (`methods`, `show_points`,
`show_bootstrap_band`, `cap_style`, `residual_box_metric`, `ylim`,
`style`, ...) pass straight through `export_huber_iqr_pdf` to each
segment's page -- including `show_points`, which switches between
"publisher" charts (the default, `show_points=False`, no raw scatter)
and "analyst" charts (`show_points=True`, individual observations
visible) without needing to fall back to a manual per-segment loop.

**`export_huber_iqr_images`** is the same function in every respect
except where the output goes: one image file per segment in a
directory, instead of one combined PDF. Same segment_cols, same
`mode`, same `title_fn`, same style parameters -- swap the function
name and `path` for `output_dir` to get individual files instead of
pages:

```python
from robustkit import export_huber_iqr_images

export_huber_iqr_images(
    df, x_col="age", y_col="salary", segment_cols=["JobFamily", "Level", "OT"],
    output_dir="salary_plots", mode="drilldown",
)
```

Useful whenever per-segment charts get consumed individually --
dropped into slide decks one at a time, browsed in a file explorer,
or fed into some other pipeline expecting separate image files --
rather than as a single document. Filenames are derived from each
segment's id or label, sanitized for the filesystem; a numeric suffix
is appended if two segments would otherwise collide. Returns the list
of saved file paths, in the order rendered.

## Loading published quantile tables (SCB / JSON-stat)

Some statistics agencies (e.g. Statistics Sweden, SCB) publish
quantiles (Q1/median/Q3) directly, with no individual-level data
available at all. `robustkit.quantiles` loads these tables generically
via JSON-stat, a standardized dimensional-data format used by SCB and
other national statistics agencies -- avoiding the fragility of
parsing metadata out of column-name strings in a wide CSV export.

```python
from robustkit import load_json_stat, plot_quantile_trend, quantile_trend_dispersion

df = load_json_stat("some_scb_table.json")

# A real SCB quirk this loader does NOT try to guess automatically:
# category labels can change meaning over time (e.g. Sweden's oldest
# working-age bracket was labeled "65-66 år" through 2022 and
# "65-68 år" from 2023, following a pension-age reform). Merge such
# cases explicitly:
df = load_json_stat(
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

## Naming conventions

A few parameter/column names look similar across the package but mean
different things -- documented here explicitly so the difference reads
as intentional, not as an inconsistency to "fix":

- **`residual` vs. `difference`:** `residual` is an INDIVIDUAL-level
  quantity (`actual - expected` for one row) -- used by
  `mad_outlier_report`, `mad_outlier_drilldown_report`, and
  `deviation_report`. `difference` is a SEGMENT/GROUP-level quantity
  (typically the median residual within a group) -- used by
  `segment_position_report`, `segment_benchmark_report`,
  `segment_benchmark_drilldown_report`, and `benchmark_report_suite`.
- **`target` vs. `y_col`:** `robustkit.information` uses `target` for
  the column being explained, since it works with arbitrary features
  (not necessarily a continuous regression outcome).
  `robustkit.benchmark`, `robustkit.segment_awareness`, and
  `robustkit.quantiles` use `y_col`, since they specifically model a
  continuous `y` as a function of `x`.
- **`segment_cols` vs. `group_columns`:** `segment_cols` (throughout
  `robustkit.segment_awareness`) is an ORDERED, most-specific-first
  list used to build a fallback HIERARCHY (see `hierarchical_segment`).
  `group_columns` (`benchmark_report_suite`) is a FLAT list of
  independent groupings, run separately with no hierarchy or fallback
  between them. Different structure, different name on purpose.
- **`min_size` vs. `min_points` vs. `min_stratum_size` vs.
  `min_group_size`:** all mean "minimum group size," but at different
  stages: `min_size` (`hierarchical_segment` and everything built on
  it) gates whether a hierarchy LEVEL gets created at all;
  `min_points` (`apply_by_segment`) gates whether an already-built
  segment gets ANALYZED; `min_stratum_size`
  (`conditional_mutual_information`) and `min_group_size`
  (`communication_score`, `rank_by_communication`) are specific to
  those `robustkit.information` calculations. Kept separate rather
  than unified to one name, since collapsing them would obscure which
  stage of a pipeline each threshold actually applies to.

## License

MIT -- see [LICENSE](LICENSE).

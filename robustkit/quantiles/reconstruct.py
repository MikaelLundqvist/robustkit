"""
Reconstruct approximate individual-level data from aggregated group
summaries (n, Q1, median, Q3) -- for the case where a statistics
agency publishes group summaries but not individual observations, and
you want to run robustkit.core analyses (Huber trend fitting, model
stability, Cook's distance, ...) as if individual data were available.

Method: calibrate a lognormal distribution so its quantiles match the
reported Q1/median/Q3 as closely as possible (via the standard
IQR-based estimator), then draw n synthetic values from it. Lognormal
is chosen because right-skewed distributions are typical for the kind
of data this module targets (salaries, and similar economic measures);
it is also the same distributional assumption already used throughout
robustkit's synthetic dataset generator, for consistency.

This complements robustkit.quantiles.trend, which handles the case
where the quantiles themselves are the final answer (a trend to plot).
This module is for when the quantiles are a STARTING point for further
analysis that needs data shaped like individual observations.

Important: because the calibration only uses three summary numbers,
the reconstructed values approximate the reported quantiles by
construction, but do NOT recover the true underlying distribution's
shape beyond what a lognormal can represent (e.g. genuine bimodality
within a group is invisible to this method). Downstream Huber/robust
fits on reconstructed data will track something close to the MEDIAN
trend the data was calibrated against, not the arithmetic mean -- see
the package README for why this is actually the intended, consistent
behavior for right-skewed data.

Also important: a lognormal's tail is unbounded and its expected
maximum grows with n -- reconstructing at a very large true n (SCB
national tables can report n in the hundreds of thousands to millions)
can produce implausibly extreme tail values. See the WARNING in
expand_aggregated_group's docstring before reconstructing at a large
true n.
"""

import numpy as np
import pandas as pd
from scipy import stats

_Z75 = stats.norm.ppf(0.75)  # ~0.6745, used to convert IQR to a lognormal sigma


def expand_aggregated_group(n, q1, median, q3, seed=0):
    """
    Draw n synthetic values from a lognormal distribution calibrated
    so its Q1/median/Q3 approximate the given values.

    Calibration:
        mu = ln(median)          (exact, since a lognormal's median is exp(mu))
        sigma = (ln(q3) - ln(q1)) / (2 * z75)   (IQR-based estimate)

    where z75 is the 75th percentile of the standard normal
    distribution. If the reported q1/median/q3 are not perfectly
    consistent with a lognormal shape (common with real data), this
    sigma is a reasonable compromise rather than an exact fit -- use
    check_reconstruction_quality to see how close the reconstruction
    actually lands.

    WARNING -- unbounded tail at large n: a lognormal distribution has
    no natural upper limit, and its expected maximum grows with n
    (roughly exp(sigma * sqrt(2 * ln(n)))). At small/moderate n (a few
    thousand) this is barely noticeable, but real SCB group sizes can
    be in the hundreds of thousands to millions (e.g. national salary
    tables) -- reconstructing at the TRUE n can produce implied
    maximum values far beyond anything that plausibly exists in
    reality (real salaries have practical ceilings a pure lognormal
    doesn't know about). This is why the examples and tests in this
    package scale n down to a few thousand for demonstration rather
    than using the true (often much larger) reported group size --
    calibration quality (matching Q1/median/Q3) does not depend on
    reproducing the true population size, but the plausibility of the
    tail does. No clipping is applied automatically; consider capping
    extreme values explicitly if you reconstruct at a large true n and
    the tail matters for your downstream analysis.
    """
    if not (q1 <= median <= q3):
        raise ValueError(f"Expected q1 <= median <= q3, got q1={q1}, median={median}, q3={q3}")
    if q1 <= 0 or median <= 0 or q3 <= 0:
        raise ValueError("expand_aggregated_group requires strictly positive values (lognormal is undefined for values <= 0)")

    rng = np.random.default_rng(seed)
    mu = np.log(median)
    sigma = 0.0 if q1 == q3 else (np.log(q3) - np.log(q1)) / (2 * _Z75)

    return rng.lognormal(mean=mu, sigma=sigma, size=int(n))


def expand_aggregated_table(df, n_col, q1_col, median_col, q3_col, group_cols=None, value_name="value", seed=0):
    """
    Expand a table of aggregated group summaries (one row per group)
    into a long-format DataFrame with one row per synthetic
    individual, calibrated to reproduce each group's reported
    quantiles.

    group_cols: columns to carry over from each aggregated row to
        every synthetic row it produces (e.g. segment, year) -- so the
        result can be fed into robustkit.core / robustkit.segmentation
        functions exactly as if it were real individual-level data.

    Each group gets an independent seed (seed + row position), so
    results are reproducible but not identical across groups.
    """
    group_cols = group_cols or []
    rows = []

    for i, row in df.reset_index(drop=True).iterrows():
        values = expand_aggregated_group(
            row[n_col], row[q1_col], row[median_col], row[q3_col], seed=seed + i,
        )
        group_data = {col: row[col] for col in group_cols}
        for v in values:
            rows.append({**group_data, value_name: v})

    return pd.DataFrame(rows)


def check_reconstruction_quality(values, q1, median, q3):
    """
    Compare the empirical quantiles of reconstructed synthetic values
    against the target quantiles they were calibrated to reproduce.

    A sanity check, not an assumption: real Q1/median/Q3 triples are
    not always perfectly consistent with a pure lognormal shape, so
    this should be checked rather than trusted blindly, especially
    before using reconstructed data for anything beyond illustration.
    """
    values = np.asarray(values, dtype=float)
    emp_q1, emp_median, emp_q3 = np.percentile(values, [25, 50, 75])

    return {
        "target": {"q1": float(q1), "median": float(median), "q3": float(q3)},
        "empirical": {"q1": float(emp_q1), "median": float(emp_median), "q3": float(emp_q3)},
        "pct_error": {
            "q1": float(abs(emp_q1 - q1) / q1 * 100),
            "median": float(abs(emp_median - median) / median * 100),
            "q3": float(abs(emp_q3 - q3) / q3 * 100),
        },
    }


# ---------------------------------------------------------------------------
# Reconstruction when only a mean is available -- no quartiles at all
# ---------------------------------------------------------------------------
#
# Some tables (e.g. SCB's age-breakdown salary tables) report only a
# mean per group, with no spread information whatsoever. Two methods
# are provided below, deliberately kept separate rather than merged
# into one "best guess" -- each makes a different, explicit assumption,
# and comparing their results (via compare_reconstruction_methods) lets
# the analyst see how much the conclusion actually depends on the
# assumption made, rather than presenting a single number as if it
# were measured rather than assumed.


def expand_aggregated_group_flat(n, mean, seed=0):
    """
    Repeat a group's mean value n times -- NO within-group spread is
    reconstructed; every synthetic observation is identical.

    This matches the original technique this package's approach is
    based on (see README): validated there for recovering regression
    COEFFICIENTS (i.e. between-group variation, such as "how much does
    the age effect differ between groups") from aggregated tables, NOT
    for representing realistic individual-level spread -- within-group
    variance is zero by construction, which understates true
    individual variation and will bias any analysis that depends on
    within-group spread (e.g. Cook's distance, dispersion measures)
    toward zero.
    """
    return np.full(int(n), float(mean), dtype=float)


def expand_aggregated_group_borrowed_dispersion(n, mean, dispersion_ratio, seed=0):
    """
    Draw n synthetic values from a lognormal distribution centered on
    `mean`, with spread calibrated from a dispersion_ratio
    ((Q3-Q1)/median) borrowed from a DIFFERENT table or population that
    does report quantiles -- for when the table at hand only reports a
    mean but a plausible spread estimate exists elsewhere.

    This stacks two separate, explicit assumptions -- treat the result
    as illustrative, not a substitute for genuine quantile data:

      1. `mean` is used as a stand-in for the median. For right-skewed
         data (e.g. salaries), the mean is systematically somewhat
         higher than the median, so this shifts the reconstructed
         center upward relative to the (unobserved) true median.
      2. `dispersion_ratio` is assumed to transfer from a different
         population -- valid only to the extent the two populations
         genuinely have similar relative spread, which is an
         assumption, not a measurement.

    Given dispersion_ratio r = (q3-q1)/median, this implies
        q1 = mean * (1 - r/2), q3 = mean * (1 + r/2)
    -- a symmetric split of the ratio around the given mean. This is
    ONE way to construct a (q1, q3) pair consistent with the ratio, not
    the only one; it is chosen for simplicity, not because it is known
    to be correct.
    """
    if dispersion_ratio <= 0:
        raise ValueError("dispersion_ratio must be positive")

    implied_q1 = mean * (1 - dispersion_ratio / 2)
    implied_q3 = mean * (1 + dispersion_ratio / 2)

    if implied_q1 <= 0:
        raise ValueError(
            f"dispersion_ratio={dispersion_ratio} implies a non-positive "
            f"Q1 ({implied_q1:.1f}) given mean={mean}; try a smaller dispersion_ratio."
        )

    return expand_aggregated_group(n=n, q1=implied_q1, median=mean, q3=implied_q3, seed=seed)


def _expand_table_generic(df, n_col, group_cols, value_name, per_row_fn):
    group_cols = group_cols or []
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        values = per_row_fn(row, i)
        group_data = {col: row[col] for col in group_cols}
        for v in values:
            rows.append({**group_data, value_name: v})
    return pd.DataFrame(rows)


def expand_aggregated_table_flat(df, n_col, mean_col, group_cols=None, value_name="value"):
    """Table-level version of expand_aggregated_group_flat -- see its docstring for the method's limitations."""
    return _expand_table_generic(
        df, n_col, group_cols, value_name,
        per_row_fn=lambda row, i: expand_aggregated_group_flat(row[n_col], row[mean_col]),
    )


def expand_aggregated_table_borrowed_dispersion(df, n_col, mean_col, dispersion_ratio, group_cols=None, value_name="value", seed=0):
    """
    Table-level version of expand_aggregated_group_borrowed_dispersion.

    dispersion_ratio: either a single float applied to every row, or
        the name of a column in df providing a per-row ratio (e.g. if
        different segments should borrow different dispersion
        estimates).
    """
    def per_row(row, i):
        ratio = row[dispersion_ratio] if isinstance(dispersion_ratio, str) else dispersion_ratio
        return expand_aggregated_group_borrowed_dispersion(row[n_col], row[mean_col], ratio, seed=seed + i)

    return _expand_table_generic(df, n_col, group_cols, value_name, per_row_fn=per_row)


def compare_reconstruction_methods(n, mean, dispersion_ratio, seed=0):
    """
    Reconstruct the same group two ways -- flat (no within-group
    spread) and borrowed-dispersion (approximate spread borrowed from
    elsewhere) -- and return both synthetic arrays plus a summary
    table, so the two sets of assumptions can be compared side by side
    rather than silently picking one and presenting it as the answer.
    """
    flat = expand_aggregated_group_flat(n, mean, seed=seed)
    borrowed = expand_aggregated_group_borrowed_dispersion(n, mean, dispersion_ratio, seed=seed)

    summary = pd.DataFrame({
        "method": ["flat", "borrowed_dispersion"],
        "mean": [float(np.mean(flat)), float(np.mean(borrowed))],
        "std": [float(np.std(flat)), float(np.std(borrowed))],
        "min": [float(np.min(flat)), float(np.min(borrowed))],
        "max": [float(np.max(flat)), float(np.max(borrowed))],
    })

    return {"flat": flat, "borrowed_dispersion": borrowed, "summary": summary}

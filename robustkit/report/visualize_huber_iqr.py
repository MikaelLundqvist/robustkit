"""
Combined view: one or more trend curves overlaid with median + IQR
error bars, plus an optional residual-quality box -- reproducing the
main diagram style used in the wage-analysis workbook this package is
based on, where the fitted model(s), the actual population spread,
and a residual-quality indicator are all shown together rather than as
separate plots.

Distinguishes itself from the other two report views:
    - plot_analyst_view shows a bootstrap CONFIDENCE band around the
      fitted curve (narrows as n grows) -- uncertainty in the estimate.
    - plot_publisher_view shows median + IQR from binned data ALONE,
      with no fitted model curve at all -- population spread only.

plot_huber_iqr shows both together: the trend curve(s) as the model's
summary, and the median + IQR as what the data actually looks like.

Two grouping strategies are supported (see `grouping`):
    - "bin" (default): quantile-based binning via dispersion_by_bin.
      Gives stable quantile estimates even in small populations, and
      works for any continuous x, not just one with few distinct
      values.
    - "unique": group by each EXACT x value (e.g. each individual age
      in years), matching a workbook-style groupby(x) aggregation.
      When a given x value has fewer than `min_n_for_iqr` observations,
      its IQR is not shown at all -- the ABSENCE of an error bar is
      itself information, signaling the sample at that exact value is
      too small to say anything about spread, rather than a plotting
      simplification. Caveat, found via validation against a real
      dataset: this only makes sense for x values with natural
      repetition (e.g. integer ages) -- for a high-precision continuous
      measurement, nearly every x value is unique and almost nothing
      meets min_n_for_iqr. Use "bin" for that case.

Style customization: `methods` selects which trend curve(s) to
overlay (default: just Huber), `style` overrides colors/linewidths/
linestyles per element, `show_bootstrap_band` adds a bootstrap
confidence band around the primary curve, `cap_style="manual"`
replaces matplotlib's default errorbar caps with hand-drawn horizontal
"hat" lines (boxplot-style), and `residual_box_metric` switches the
info box between R^2/MAE/RMSE and MdAPE/IQR(resid). All of these
default to the package's existing plain style -- nothing changes
unless explicitly requested, so this is a strict extension of the
previous, simpler plot_huber_iqr.
"""

import numpy as np
import pandas as pd

from ..core.trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend
from ..core.goodness_of_fit import goodness_of_fit
from ..core.uncertainty import bootstrap_band
from .dispersion import dispersion_by_bin


_METHOD_FITTERS = {
    "huber": fit_huber_trend,
    "tukey": fit_tukey_trend,
    "ols": fit_ols_trend,
}

# Default per-method line style, matching a common four-curve robust-
# analysis convention (solid Huber, dashed Tukey, dotted OLS, dash-dot
# median ensemble) -- override via the `style` argument.
_DEFAULT_METHOD_STYLE = {
    "huber": {"color": "steelblue", "linewidth": 2, "linestyle": "-", "label": "Huber trend"},
    "tukey": {"color": "blue", "linewidth": 2, "linestyle": "--", "label": "Tukey trend"},
    "ols": {"color": "orange", "linewidth": 2, "linestyle": ":", "label": "OLS trend"},
    "median_ensemble": {"color": "green", "linewidth": 2, "linestyle": "-.", "label": "Median ensemble"},
}

_DEFAULT_STYLE = {
    "iqr_color": "darkorange",
    "iqr_elinewidth": 1.5,
    "iqr_capsize": 4,            # used only when cap_style="matplotlib"
    "iqr_marker_size": None,     # None -> matplotlib default marker size for errorbar 'o'
    "cap_width": 0.15,           # in x units, used only when cap_style="manual"
    "cap_linewidth": 1.0,        # used only when cap_style="manual"
    "points_color": "gray",
    "points_alpha": 0.15,
    "points_size": 10,
    "bootstrap_color": "gray",
    "bootstrap_alphas": {95: 0.15, 50: 0.30},
    "grid_linestyle": "--",
    "grid_alpha": 0.3,
}


def _merge_style(defaults, overrides):
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def _unique_value_summary(x, y, min_n_for_iqr=5):
    """
    Median/Q1/Q3 grouped by each EXACT x value (not binned). Q1/Q3 are
    NaN for any x value with fewer than min_n_for_iqr observations --
    intentionally, so the caller can distinguish "spread is small" from
    "spread is unknown because the sample is too small to say."
    """
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})

    rows = []
    for x_val, group in df.groupby("x"):
        n = len(group)
        median = float(group["y"].median())
        if n >= min_n_for_iqr:
            q1 = float(group["y"].quantile(0.25))
            q3 = float(group["y"].quantile(0.75))
        else:
            q1 = np.nan
            q3 = np.nan
        rows.append({"x_center": x_val, "n": n, "q1": q1, "median": median, "q3": q3})

    return pd.DataFrame(rows).sort_values("x_center").reset_index(drop=True)


def _median_ensemble_curve(x, y, degree, grid):
    """
    Pointwise median of Huber/Tukey/OLS predictions across the grid --
    a simple, robust "consensus" curve. Requires all three methods to
    be fittable (Tukey needs statsmodels).
    """
    curves = []
    for method in ("huber", "tukey", "ols"):
        fit = _METHOD_FITTERS[method](x, y, degree=degree)
        curves.append(predict_trend(fit, grid))
    return np.median(np.vstack(curves), axis=0)


def plot_huber_iqr(x, y, degree=2, bins=15, grouping="bin", min_n_for_iqr=5,
                    show_points=False, show_residual_box=True, residual_box_metric="r2",
                    methods=("huber",), show_bootstrap_band=False, bootstrap_levels=(95,),
                    n_boot="auto", cap_style="matplotlib", ylim="auto",
                    style=None, title=None, ax=None, figsize=(10, 6)):
    """
    Plot one or more trend curves together with median + IQR error bars.

    methods: which trend curve(s) to overlay, drawn in this order:
        any of "huber", "tukey", "ols", "median_ensemble" (the
        pointwise median of the other three). Default ("huber",)
        matches the original, simpler behavior of this function.
        "tukey" and "median_ensemble" require statsmodels.

    show_bootstrap_band: if True, adds a bootstrap confidence band
        (via bootstrap_band) around the FIRST method in `methods`.
        bootstrap_levels: e.g. (95,) or (95, 50) for two nested bands
        at different alpha, matching a common "outer 95% / inner 50%"
        presentation style.

    cap_style: "matplotlib" (default) uses errorbar's built-in caps.
        "manual" instead draws short horizontal lines at each Q1/Q3
        endpoint (boxplot-style "hats"), matching a common manual
        salary-chart convention -- width controlled by
        style={"cap_width": ...}.

    residual_box_metric: "r2" (default) shows R^2/MAE/RMSE of the
        first method in `methods`. "mdape" instead shows median
        absolute percentage error and IQR of residuals -- a more
        HR-report-friendly framing ("typical error is X%", "spread of
        errors is Y kr") than a statistical R^2.

    ylim: "auto" (default) leaves matplotlib's automatic limits.
        Pass "dynamic" to set limits from the data itself (min*0.95,
        max*1.05, matching a common salary-chart convention that avoids
        curves touching the plot edges), or an explicit (lo, hi) tuple.

    style: dict overriding any of the default style values (colors,
        line widths, alphas, cap width, ...) -- see _DEFAULT_STYLE and
        _DEFAULT_METHOD_STYLE in this module for all overridable keys.
        Only keys you provide are changed; everything else keeps its
        default.

    grouping, min_n_for_iqr, show_points, show_residual_box, title, ax,
    figsize, bins: unchanged from the original version -- see the
    module docstring for grouping semantics.

    Returns a dict with `grid`, one prediction array per requested
    method (keyed by method name), and `binned` (the per-bin or
    per-unique-x summary DataFrame).
    """
    import matplotlib.pyplot as plt

    if grouping not in ("bin", "unique"):
        raise ValueError(f"grouping must be 'bin' or 'unique', got {grouping!r}")
    if residual_box_metric not in ("r2", "mdape"):
        raise ValueError(f"residual_box_metric must be 'r2' or 'mdape', got {residual_box_metric!r}")
    if cap_style not in ("matplotlib", "manual"):
        raise ValueError(f"cap_style must be 'matplotlib' or 'manual', got {cap_style!r}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = _merge_style(_DEFAULT_STYLE, style)

    grid = np.linspace(x.min(), x.max(), 200)
    curves = {}
    for method in methods:
        if method == "median_ensemble":
            curves[method] = _median_ensemble_curve(x, y, degree, grid)
        else:
            fit = _METHOD_FITTERS[method](x, y, degree=degree)
            curves[method] = predict_trend(fit, grid)

    if grouping == "bin":
        binned = dispersion_by_bin(x, y, n_bins=bins)
    else:
        binned = _unique_value_summary(x, y, min_n_for_iqr=min_n_for_iqr)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.scatter(x, y, s=s["points_size"], alpha=s["points_alpha"], color=s["points_color"], zorder=1)

    if show_bootstrap_band:
        primary_method = methods[0]
        for level in sorted(bootstrap_levels, reverse=True):
            band = bootstrap_band(x, y, degree=degree, n_boot=n_boot, ci=level)
            alpha = s["bootstrap_alphas"].get(level, 0.2)
            ax.fill_between(
                band["grid"], band["lower"], band["upper"],
                color=s["bootstrap_color"], alpha=alpha, zorder=1,
                label=f"{level}% bootstrap ({primary_method})",
            )

    has_iqr = binned["q1"].notna() & binned["q3"].notna()
    with_iqr = binned[has_iqr]
    without_iqr = binned[~has_iqr]

    if len(with_iqr) > 0:
        yerr_lower = with_iqr["median"] - with_iqr["q1"]
        yerr_upper = with_iqr["q3"] - with_iqr["median"]
        iqr_label = (
            f"Median (IQR error bars, n\u2265{min_n_for_iqr})" if grouping == "unique"
            else "Median per bin (IQR error bars)"
        )
        errorbar_kwargs = dict(
            fmt="o", color=s["iqr_color"], ecolor=s["iqr_color"], elinewidth=s["iqr_elinewidth"],
            label=iqr_label, zorder=3,
        )
        if s["iqr_marker_size"] is not None:
            errorbar_kwargs["markersize"] = s["iqr_marker_size"]
        if cap_style == "matplotlib":
            errorbar_kwargs["capsize"] = s["iqr_capsize"]
        else:
            errorbar_kwargs["capsize"] = 0

        ax.errorbar(with_iqr["x_center"], with_iqr["median"], yerr=[yerr_lower, yerr_upper], **errorbar_kwargs)

        if cap_style == "manual":
            cap_width = s["cap_width"]
            for x_c, q1, q3 in zip(with_iqr["x_center"], with_iqr["q1"], with_iqr["q3"]):
                ax.plot([x_c - cap_width, x_c + cap_width], [q1, q1], color=s["iqr_color"], linewidth=s["cap_linewidth"], zorder=3)
                ax.plot([x_c - cap_width, x_c + cap_width], [q3, q3], color=s["iqr_color"], linewidth=s["cap_linewidth"], zorder=3)

    if len(without_iqr) > 0:
        ax.scatter(
            without_iqr["x_center"], without_iqr["median"],
            marker="o", facecolors="none", edgecolors=s["iqr_color"], s=40,
            label=f"Median (n<{min_n_for_iqr}, spread not shown)", zorder=3,
        )

    for method in methods:
        method_style = _merge_style(_DEFAULT_METHOD_STYLE.get(method, {}), (style or {}).get(f"{method}_line"))
        ax.plot(
            grid, curves[method],
            color=method_style.get("color", "steelblue"),
            linewidth=method_style.get("linewidth", 2),
            linestyle=method_style.get("linestyle", "-"),
            label=method_style.get("label", method),
            zorder=2,
        )

    if show_residual_box:
        primary_method = methods[0]
        if residual_box_metric == "r2":
            gof = goodness_of_fit(x, y, degree=degree, method=primary_method if primary_method in ("huber", "tukey", "ols") else "huber")
            textstr = f"R\u00b2={gof['r_squared']:.2f}\nMAE={gof['mae']:.0f}\nRMSE={gof['rmse']:.0f}"
        else:
            fit_for_metric = _METHOD_FITTERS.get(primary_method, fit_huber_trend)(x, y, degree=degree)
            pred = predict_trend(fit_for_metric, x)
            residuals = y - pred
            ape = np.abs(residuals / np.where(y != 0, y, np.nan)) * 100
            mdape = float(np.nanmedian(ape))
            iqr_resid = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
            textstr = f"MdAPE: {mdape:.1f} %\nIQR(resid): {iqr_resid:,.0f}".replace(",", " ")
        ax.text(
            0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    if ylim == "dynamic":
        y_all = [np.asarray(v) for v in curves.values()]
        if len(with_iqr) > 0:
            y_all.append(with_iqr["median"].to_numpy())
            y_all.append(with_iqr["q1"].to_numpy())
            y_all.append(with_iqr["q3"].to_numpy())
        y_concat = np.concatenate([np.atleast_1d(a) for a in y_all])
        ax.set_ylim(y_concat.min() * 0.95, y_concat.max() * 1.05)
    elif ylim not in (None, "auto"):
        ax.set_ylim(*ylim)

    ax.set_title(title if title is not None else "Huber trend with per-bin median and IQR")
    ax.legend(loc="best")
    ax.grid(True, linestyle=s["grid_linestyle"], alpha=s["grid_alpha"])

    if created_fig:
        fig.tight_layout()

    return {"grid": grid, **curves, "binned": binned}

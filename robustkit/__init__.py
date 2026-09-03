"""
robustkit
=========

Practical tools for robust analysis of a single continuous relationship
(y as a function of one continuous x), designed around one core idea:
a conclusion that survives multiple fitting methods is more trustworthy
than one that only holds under a single model.

Core building blocks (robustkit.core):
    - Trend fitting: Huber, Tukey biweight, and OLS
    - Stability: how much does the fitted curve change across methods?
    - Diagnostics: which points are influential, and how much does it matter?
    - Uncertainty: percentile and BCa bootstrap confidence bands

All core functions operate on a single (x, y) pair. Segmentation (running
the same analysis across subgroups of a larger dataset) is a separate,
composable layer -- see robustkit.segmentation.
"""

from .core.trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend
from .core.stability import model_stability_pct
from .core.diagnostics import cooks_diagnostic, cook_impact
from .core.uncertainty import bootstrap_band, bca_bootstrap_ci
from .core.consistency import check_row_integrity, compare_row_sets
from .segmentation.hierarchy import hierarchical_segment, segment_sizes
from .segmentation.apply import apply_by_segment

__all__ = [
    "fit_huber_trend",
    "fit_tukey_trend",
    "fit_ols_trend",
    "predict_trend",
    "model_stability_pct",
    "cooks_diagnostic",
    "cook_impact",
    "bootstrap_band",
    "bca_bootstrap_ci",
    "check_row_integrity",
    "compare_row_sets",
    "hierarchical_segment",
    "segment_sizes",
    "apply_by_segment",
]

__version__ = "0.1.0"

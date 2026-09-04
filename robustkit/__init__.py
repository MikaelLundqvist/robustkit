"""
robustkit
=========

Practical tools for robust analysis of a single continuous relationship
(y as a function of one continuous x), designed around one core idea:
a conclusion that survives multiple fitting methods is more trustworthy
than one that only holds under a single model.

Modules:
    robustkit.core          -- trend fitting, stability, diagnostics, uncertainty
    robustkit.segmentation   -- hierarchical grouping, per-segment analysis
    robustkit.information    -- mutual-information-based feature ranking
"""

from .core.trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend
from .core.stability import model_stability_pct
from .core.diagnostics import cooks_diagnostic, cook_impact
from .core.uncertainty import bootstrap_band, bca_bootstrap_ci
from .core.consistency import check_row_integrity, compare_row_sets
from .segmentation.hierarchy import hierarchical_segment, segment_sizes
from .segmentation.apply import apply_by_segment
from .information.entropy import entropy
from .information.mutual_info import rank_features, information_efficiency
from .information.quadrants import quadrant_report
from .information.visualization import plot_feature_space, feature_map
from .information.profile import profile, print_profile

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
    "entropy",
    "rank_features",
    "information_efficiency",
    "quadrant_report",
    "plot_feature_space",
    "feature_map",
    "profile",
    "print_profile",
]

__version__ = "0.0.1"

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

from .core.trend import fit_huber_trend, fit_tukey_trend, fit_ols_trend, predict_trend, trend_derivative
from .core.stability import model_stability_pct
from .core.diagnostics import cooks_diagnostic, cook_impact
from .core.uncertainty import bootstrap_band, bca_bootstrap_ci
from .core.consistency import check_row_integrity, compare_row_sets
from .core.goodness_of_fit import goodness_of_fit, compare_polynomial_degrees
from .segmentation.hierarchy import hierarchical_segment, segment_sizes
from .segmentation.apply import apply_by_segment
from .information.entropy import entropy
from .information.mutual_info import rank_features, information_efficiency
from .information.quadrants import quadrant_report
from .information.visualization import plot_feature_space, feature_map
from .information.profile import profile, print_profile
from .information.conditional_mi import conditional_mutual_information
from .information.communication import communication_score, rank_by_communication
from .information.pairs import pair_redundancy, pair_synergy, rank_communicative_pairs
from .common.quadrants import classify_quadrants
from .benchmark.global_model import fit_huber_benchmark, segment_position_report
from .benchmark.robustness_map import feature_robustness_report, plot_feature_robustness
from .report.dispersion import iqr, dispersion_ratio, dispersion_by_bin
from .report.visualize_analyst import plot_analyst_view
from .report.visualize_publisher import plot_publisher_view
from .quantiles.io import load_scb_json_stat
from .quantiles.trend import prepare_quantile_trend, plot_quantile_trend, quantile_trend_dispersion
from .quantiles.reconstruct import (
    expand_aggregated_group, expand_aggregated_table, check_reconstruction_quality,
    expand_aggregated_group_flat, expand_aggregated_group_borrowed_dispersion,
    expand_aggregated_table_flat, expand_aggregated_table_borrowed_dispersion,
    compare_reconstruction_methods,
)

__all__ = [
    "fit_huber_trend",
    "fit_tukey_trend",
    "fit_ols_trend",
    "predict_trend",
    "trend_derivative",
    "model_stability_pct",
    "cooks_diagnostic",
    "cook_impact",
    "bootstrap_band",
    "bca_bootstrap_ci",
    "check_row_integrity",
    "compare_row_sets",
    "goodness_of_fit",
    "compare_polynomial_degrees",
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
    "conditional_mutual_information",
    "communication_score",
    "rank_by_communication",
    "pair_redundancy",
    "pair_synergy",
    "rank_communicative_pairs",
    "classify_quadrants",
    "fit_huber_benchmark",
    "segment_position_report",
    "feature_robustness_report",
    "plot_feature_robustness",
    "iqr",
    "dispersion_ratio",
    "dispersion_by_bin",
    "plot_analyst_view",
    "plot_publisher_view",
    "load_scb_json_stat",
    "prepare_quantile_trend",
    "plot_quantile_trend",
    "quantile_trend_dispersion",
    "expand_aggregated_group",
    "expand_aggregated_table",
    "check_reconstruction_quality",
    "expand_aggregated_group_flat",
    "expand_aggregated_group_borrowed_dispersion",
    "expand_aggregated_table_flat",
    "expand_aggregated_table_borrowed_dispersion",
    "compare_reconstruction_methods",
]

__version__ = "0.0.1"

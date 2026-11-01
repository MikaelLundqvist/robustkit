# Changelog

## 0.7.0

Release covering everything since v0.6.0: dual-reference outlier
classification, a reference parameter for the outlier reports, benchmark
screening and goodness-of-fit, per-segment Huber+IQR export, pointwise
stability, and a set of drilldown and quality reports. Existing v0.6.0
calls keep working; every new parameter has a default that reproduces
the old behavior.

### robustkit.segment_awareness
- mad_outlier_report, mad_outlier_drilldown_report, export_outlier_pdf:
  new reference parameter. "model" (default, unchanged) compares each
  person to the benchmark's own prediction; "curve" fits a fresh Huber
  trend over x_col within each segment; "benchmark_curve" fits that
  trend to the benchmark's predictions. A segment's curve keeps the
  benchmark's information for every variable the segmentation holds
  constant (see Chapter 13).
- export_outlier_pdf: new annotate_flagged_with, mode, segment_mode,
  reference and title_fn parameters.
- dual_reference_outlier_report / dual_reference_outlier_drilldown_report:
  classify each individual against two reference populations at once
  (Type A/B/C/D), the latter at every hierarchy level.
- segment_stability_drilldown_report, outlier_drilldown_summary (number of
  hierarchy levels on which each individual is flagged),
  segment_quality_report (stability, Cook's impact, MdAPE and residual IQR
  in one row per segment), segment_contribution_report (structural vs
  local effect of a segment's difference).
- export_huber_iqr_pdf / export_huber_iqr_images: one plot_huber_iqr
  chart per segment, as a PDF or as individual image files.

### robustkit.benchmark
- benchmark_goodness_of_fit: r_squared, rmse and mae of an already-fitted
  benchmark.
- benchmark_screening_report: screens a fitted benchmark for likely
  missing structure without fitting a new model.
- plot_segment_vs_benchmark: where along x_col a segment diverges from
  the benchmark.

### robustkit.core
- pointwise_stability_report / plot_pointwise_stability: spread between
  the Huber, Tukey and OLS curves against the Huber bootstrap band, per
  point on a grid over x.
- bca_bootstrap_ci_by_index, segment_position_report,
  segment_benchmark_drilldown_report: new jackknife_cap (default 1000).
  Above that many observations the acceleration term is estimated from a
  random subsample, which keeps large segments fast.

### robustkit.report
- plot_huber_iqr: new show_undersized_points (default True).

### robustkit.quantiles
- reconstruct: documentation of what the reconstruction does and does not
  provide (n-weighting, means vs medians, scaling very large n, and the
  mean-vs-median bias of a lognormal reconstruction). Documentation only.

### Packaging
- __version__ now matches pyproject.toml (it read 0.0.1 in v0.6.0).
- publish workflow: can also be started manually from the Actions tab.
- 14 new test modules.

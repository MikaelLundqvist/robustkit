"""
quantiles_suite -- rigorous validation of robustkit.quantiles
(io, trend, reconstruct) against both synthetic edge cases and real
SCB JSON-stat exports.

Structure mirrors robustkit.quantiles itself:
    run_io.py           -- load_scb_json_stat
    run_trend.py         -- prepare_quantile_trend, plot_quantile_trend, quantile_trend_dispersion
    run_reconstruct.py   -- expand_aggregated_group/_table, check_reconstruction_quality,
                            and the mean-only (flat / borrowed_dispersion) variants

Real-data checks look for SCB JSON-stat exports in
validation/quantiles_suite/data/ (quartiles.json, age.json). These
files are NOT included in the repository (place your own SCB exports
there); if absent, real-data checks are skipped with a clear message
and edge-case checks (which need no external data) still run.

Run the full suite with:

    python -m validation.quantiles_suite.main
"""

"""
robustkit.segment_awareness
============================

Convenience wrappers that build a hierarchical segmentation
automatically from a flat list of grouping columns, then run an
existing robustkit analysis (model_stability_pct, segment_position_report)
within the resulting segments -- so segmenting a population is not a
separate preparation step the user has to write out by hand before
every analysis.

This module adds no new statistical logic: everything here is a thin
composition of robustkit.segmentation.hierarchical_segment with
robustkit.core / robustkit.benchmark functions that already exist.
The point is convenience and a consistent fallback order, not new
capability.
"""

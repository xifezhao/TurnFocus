"""Evaluation metrics, online engine, and ablation/sweep exports."""

from turnfocus.evaluation.metrics import (
    cohens_kappa_binary,
    compute_temporal_grounding_iou,
    mean_and_95ci,
    summarize_latency_percentiles,
    summarize_lp_optimality_distribution,
)
from turnfocus.evaluation.pipeline import TurnFocusEngine, TurnFocusHyperparams
from turnfocus.evaluation.ablations_and_sweeps import build_ablation_variants, build_sensitivity_sweep_grid

__all__ = [
    "TurnFocusEngine",
    "TurnFocusHyperparams",
    "build_ablation_variants",
    "build_sensitivity_sweep_grid",
    "cohens_kappa_binary",
    "compute_temporal_grounding_iou",
    "mean_and_95ci",
    "summarize_latency_percentiles",
    "summarize_lp_optimality_distribution",
]

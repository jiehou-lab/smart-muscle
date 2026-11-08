"""Evaluation: metrics, patch-to-slide aggregation, artefact writing."""

from .aggregate import aggregate_to_wsi, wsi_metrics
from .metrics import MetricSet, compute_metrics, find_optimal_threshold, ovr_roc, predict_labels
from .rescore import rescore_predictions
from .runner import EvaluationResult, evaluate_experiment, evaluate_predictions_csv, predict_patches

__all__ = [
    "MetricSet",
    "compute_metrics",
    "predict_labels",
    "ovr_roc",
    "find_optimal_threshold",
    "aggregate_to_wsi",
    "wsi_metrics",
    "EvaluationResult",
    "evaluate_experiment",
    "evaluate_predictions_csv",
    "rescore_predictions",
    "predict_patches",
]

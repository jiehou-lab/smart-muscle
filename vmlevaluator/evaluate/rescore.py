"""Re-score an existing predictions table, without loading any model.

Kept deliberately free of torch imports: the figure notebooks only need
``pandas``, ``numpy`` and ``scikit-learn`` to turn archived predictions back
into metrics, so regenerating the manuscript figures never requires a deep
learning stack or a GPU.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from .aggregate import aggregate_to_wsi, wsi_metrics
from .metrics import MetricSet, compute_metrics, predict_labels

__all__ = ["prob_columns", "rescore_predictions"]


def prob_columns(df: pd.DataFrame) -> list:
    return sorted(
        [c for c in df.columns if c.startswith("predicted_prob_class_")],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )


def rescore_predictions(
    predictions_csv,
    *,
    task_type: str = "binary",
    class_names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    recompute_labels: bool = False,
) -> Dict[str, object]:
    """Patch- and WSI-level metrics from a ``predictions.csv``.

    ``recompute_labels`` re-derives hard labels from the stored probabilities
    instead of trusting the ``predicted_label`` column — useful when checking
    that an archived run used the 0.5 threshold it claims to have used.
    """
    df = pd.read_csv(predictions_csv)
    cols = prob_columns(df)
    if not cols:
        raise ValueError(f"{predictions_csv} has no predicted_prob_class_* columns")

    y_true = df["true_label"].to_numpy()
    y_proba = df[cols].to_numpy()
    if recompute_labels or "predicted_label" not in df.columns:
        y_pred = predict_labels(y_proba, task_type, threshold)
    else:
        y_pred = df["predicted_label"].to_numpy()

    patch: MetricSet = compute_metrics(
        y_true, y_pred, y_proba, task_type=task_type, class_names=class_names, level="patch"
    )
    wsi_table = aggregate_to_wsi(df, task_type=task_type, threshold=threshold)
    wsi: MetricSet = wsi_metrics(wsi_table, task_type=task_type, class_names=class_names)
    return {"patch": patch, "wsi": wsi, "wsi_table": wsi_table, "predictions": df}

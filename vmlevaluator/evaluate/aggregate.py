"""Patch-to-slide aggregation.

Labels exist at the whole-slide level, so a specimen-level decision is formed
by averaging patch probabilities within a slide::

    p_WSI = (1 / N) * sum_i p_i

and then thresholding (binary) or taking the argmax (multiclass). This is the
aggregation rule stated in the manuscript's Slide-Level Inference section.

Why it matters: patches from one slide share a label and, for multimodal
models, identical morphometrics, so patch-level observations are not
independent. WSI-level numbers are the ones that speak to specimen-level
generalisation — with the caveat that the rat test splits contain only 8, 6
and 11 slides.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.wsi import extract_wsi_id
from .metrics import MetricSet, compute_metrics, predict_labels

__all__ = ["aggregate_to_wsi", "wsi_metrics", "PROB_PREFIX"]

PROB_PREFIX = "predicted_prob_class_"


def _prob_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[-1]))


def aggregate_to_wsi(
    predictions: pd.DataFrame,
    *,
    task_type: str = "binary",
    threshold: float = 0.5,
    filename_column: str = "filename",
) -> pd.DataFrame:
    """Collapse a patch-level predictions table to one row per slide.

    ``predictions`` needs ``filename``, ``true_label`` and
    ``predicted_prob_class_*`` columns — the schema written by
    :mod:`vmlevaluator.evaluate.runner` and by the published evaluation runs.
    """
    prob_cols = _prob_columns(predictions)
    if not prob_cols:
        raise ValueError(
            f"No '{PROB_PREFIX}*' columns found in the predictions table; "
            f"got {list(predictions.columns)}"
        )

    df = predictions.copy()
    df["wsi_id"] = df[filename_column].map(extract_wsi_id)

    grouped = df.groupby("wsi_id", sort=False)
    mean_probs = grouped[prob_cols].mean()
    true_labels = grouped["true_label"].first()
    n_patches = grouped.size().rename("n_patches")

    out = mean_probs.join(true_labels).join(n_patches).reset_index()
    out["predicted_label"] = predict_labels(
        out[prob_cols].to_numpy(), task_type=task_type, threshold=threshold
    )
    return out


def wsi_metrics(
    wsi_table: pd.DataFrame,
    *,
    task_type: str = "binary",
    class_names: Optional[Sequence[str]] = None,
) -> MetricSet:
    """Metrics over the aggregated slide table."""
    prob_cols = _prob_columns(wsi_table)
    return compute_metrics(
        wsi_table["true_label"].to_numpy(),
        wsi_table["predicted_label"].to_numpy(),
        wsi_table[prob_cols].to_numpy(),
        task_type=task_type,
        class_names=class_names,
        level="wsi",
    )

"""Metric computation for patch-level and slide-level predictions.

Conventions follow the manuscript's Evaluation Metrics section:

* **binary tasks** report precision / recall / F1 / ROC-AUC for the *positive*
  class (injured, or the higher-severity class), thresholding the class-1
  probability at 0.5;
* **multiclass severity grading** reports macro-averaged precision / recall /
  F1 and one-vs-rest macro ROC-AUC, predicting by argmax.

Every function takes plain arrays so it can be reused for patch-level,
WSI-level and cross-species predictions alike.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "MetricSet",
    "predict_labels",
    "compute_metrics",
    "binary_roc",
    "ovr_roc",
    "find_optimal_threshold",
]

DEFAULT_THRESHOLD = 0.5


@dataclass
class MetricSet:
    """Metrics for one prediction set, at one aggregation level."""

    level: str                     #: "patch" or "wsi"
    task_type: str                 #: "binary" or "multiclass"
    n: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: Optional[float] = None
    auc: Optional[float] = None            #: binary ROC-AUC
    macro_auc: Optional[float] = None      #: multiclass one-vs-rest macro AUC
    pos_precision: Optional[float] = None
    pos_recall: Optional[float] = None
    pos_f1: Optional[float] = None
    pos_class: Optional[str] = None
    per_class_f1: Optional[List[float]] = None
    #: One-vs-rest metrics for the most injured class. On a binary task these
    #: are identical to the pos_* fields, because the classes are severity
    #: ordered and the positive class *is* the most injured one. On the
    #: three-class task they answer "how well is Very poor found?", which
    #: macro-F1 averages away.
    most_injured_class: Optional[str] = None
    most_injured_f1: Optional[float] = None
    most_injured_precision: Optional[float] = None
    most_injured_recall: Optional[float] = None
    most_injured_auc: Optional[float] = None

    @property
    def headline_f1(self) -> float:
        """The F1 the manuscript figures plot: positive-class, or macro."""
        if self.task_type == "binary" and self.pos_f1 is not None:
            return self.pos_f1
        return self.macro_f1

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["headline_f1"] = self.headline_f1
        return d


def predict_labels(
    y_proba: np.ndarray, task_type: str, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Probabilities to hard labels: threshold for binary, argmax for multiclass."""
    y_proba = np.asarray(y_proba)
    if task_type == "binary" and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return (y_proba[:, 1] > threshold).astype(np.int64)
    return y_proba.argmax(axis=1).astype(np.int64)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
    *,
    task_type: str = "binary",
    class_names: Optional[Sequence[str]] = None,
    level: str = "patch",
    pos_label: int = 1,
) -> MetricSet:
    """Compute the full metric set for one prediction array."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = 2 if task_type == "binary" else int(max(2, len(class_names or []) or (y_true.max() + 1)))

    ms = MetricSet(
        level=level,
        task_type=task_type,
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class_f1=[
            float(v)
            for v in f1_score(
                y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
            )
        ],
    )

    if task_type == "binary":
        ms.pos_precision = float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )
        ms.pos_recall = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        ms.pos_f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        if class_names is not None and pos_label < len(class_names):
            ms.pos_class = str(class_names[pos_label])
        if y_proba is not None and len(np.unique(y_true)) > 1:
            scores = np.asarray(y_proba)
            scores = scores[:, pos_label] if scores.ndim == 2 else scores
            ms.auc = float(roc_auc_score(y_true, scores))
    else:
        if y_proba is not None:
            _, macro_auc, _ = ovr_roc(y_true, np.asarray(y_proba), n_classes, class_names)
            ms.macro_auc = macro_auc

    # One-vs-rest for the most injured class. Classes are severity ordered, so
    # that is always the last index.
    injured = n_classes - 1
    ms.most_injured_f1 = float(
        f1_score(y_true == injured, y_pred == injured, zero_division=0))
    ms.most_injured_precision = float(
        precision_score(y_true == injured, y_pred == injured, zero_division=0))
    ms.most_injured_recall = float(
        recall_score(y_true == injured, y_pred == injured, zero_division=0))
    if class_names is not None and injured < len(class_names):
        ms.most_injured_class = str(class_names[injured])
    if y_proba is not None and len(np.unique(y_true == injured)) > 1:
        scores = np.asarray(y_proba)
        if scores.ndim == 2 and scores.shape[1] > injured:
            ms.most_injured_auc = float(roc_auc_score(y_true == injured, scores[:, injured]))
        elif scores.ndim == 1 and injured == 1:
            ms.most_injured_auc = float(roc_auc_score(y_true == injured, scores))

    return ms


def binary_roc(y_true, y_score):
    """``(fpr, tpr, auc)`` for the positive class, or ``None`` if degenerate."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score), pos_label=1)
    return fpr, tpr, float(roc_auc_score(y_true, y_score))


def ovr_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
):
    """One-vs-rest ROC per class plus macro / weighted OvR AUC.

    Returns ``(curves, macro_auc, weighted_auc)`` where ``curves`` maps class
    index to ``{name, auc, fpr, tpr}``. A class absent from ``y_true`` yields
    ``auc=None`` rather than raising, which happens on the small WSI-level
    test splits.
    """
    y_true = np.asarray(y_true)
    curves: Dict[int, Dict] = {}
    labels = list(range(num_classes))

    for i in labels:
        name = str(class_names[i]) if class_names is not None and i < len(class_names) else f"class_{i}"
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            curves[i] = {"name": name, "auc": None, "fpr": None, "tpr": None}
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        curves[i] = {
            "name": name,
            "auc": float(roc_auc_score(y_bin, y_proba[:, i])),
            "fpr": fpr,
            "tpr": tpr,
        }

    macro_auc = weighted_auc = None
    for average, target in (("macro", "macro"), ("weighted", "weighted")):
        try:
            value = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average=average, labels=labels)
            )
        except ValueError:
            value = None
        if target == "macro":
            macro_auc = value
        else:
            weighted_auc = value

    return curves, macro_auc, weighted_auc


def find_optimal_threshold(y_true, y_proba, metric: str = "f1", pos_label: int = 1):
    """Threshold maximising ``f1`` or Youden's J on the ROC curve.

    Reported metrics in the manuscript use the fixed 0.5 threshold; this is a
    diagnostic for the notebooks, not part of the published pipeline.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(y_proba)
    probs = probs[:, pos_label] if probs.ndim == 2 else probs

    fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=pos_label)
    best_threshold, best_score = DEFAULT_THRESHOLD, -1.0
    for idx, threshold in enumerate(thresholds):
        pred = (probs >= threshold).astype(int)
        score = (
            f1_score(y_true, pred, pos_label=pos_label, zero_division=0)
            if metric == "f1"
            else tpr[idx] - fpr[idx]
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def text_report(y_true, y_pred, class_names: Sequence[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=list(class_names), zero_division=0
    )

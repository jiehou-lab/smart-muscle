"""Shared constants: batch keys, task registry, class names.

Batch-dict keys are kept identical to the original code base
(``CNN_only/const.py`` and ``cnn_with_numerical_values/const.py``) so that
checkpoints and downstream code remain interchangeable.
"""

from __future__ import annotations

from typing import Dict, List

from .labels import (FOLDER_TO_CLASS, MOST_INJURED_INDEX,  # noqa: F401
                     POSITIVE_CLASS_INDEX, SEVERITY_CLASS_NAMES)

# ── Batch dictionary keys ────────────────────────────────────────────────────
IMAGE = "image"
LABEL = "label"
NUMERICAL = "numerical"

# ── Output keys ──────────────────────────────────────────────────────────────
LOGITS = "logits"
FEATURES = "features"

# ── Morphometric feature names (slide level) ─────────────────────────────────
MORPHOMETRIC_FEATURES: List[str] = ["avg_csa", "num_fibers"]

# ── Task registry ────────────────────────────────────────────────────────────
#: Canonical task keys used throughout the package, the CLI and the notebooks.
TASKS: List[str] = ["control_vml", "poor_very_poor", "multiclass"]

TASK_LABELS: Dict[str, str] = {
    "control_vml": "Control vs VML",
    "poor_very_poor": "Poor vs Very poor",
    "multiclass": "Multiclass (Perfect / Poor / Very poor)",
}

# Class names in severity-increasing order. The folder integers on disk are
# permuted onto these indices as the data is read -- see vmlevaluator.labels,
# which holds the permutation and the evidence behind it. The consequence worth
# remembering: **class index increases with injury, so the positive class is
# always the most injured one**.
TASK_CLASS_NAMES: Dict[str, List[str]] = dict(SEVERITY_CLASS_NAMES)

TASK_NUM_CLASSES: Dict[str, int] = {
    "control_vml": 2,
    "poor_very_poor": 2,
    "multiclass": 3,
}

#: Model keys understood by :func:`vmlevaluator.config.load_experiment`.
#:
#: ``rf`` / ``rf_morph`` are the classical tree-ensemble pipeline (flattened
#: pixels, optionally + morphometrics). The manuscript's XGBoost models run
#: through it with the estimator swapped (``scripts/run_xgb.py``).
MODELS: List[str] = [
    "lr",          # logistic regression, flattened pixels
    "lr_morph",    # logistic regression, pixels + morphometrics
    "rf",          # classical tree pipeline, flattened pixels (XGBoost)
    "rf_morph",    # classical tree pipeline, pixels + morphometrics (XGBoost + morph)
    "cnn",         # ResNet-50, image only
    "cnn_morph",   # ResNet-50 + morphometric encoder (fusion)
    "mlp_morph",   # morphometric-only MLP (Table 2 ablation)
]

MODEL_LABELS: Dict[str, str] = {
    "lr": "LR",
    "lr_morph": "LR with morphometric",
    "rf": "RF",
    "rf_morph": "RF with morphometric",
    "cnn": "ResNet50",
    "cnn_morph": "ResNet50 with morphometric",
    "mlp_morph": "MLP (morphometric only)",
}

#: Model families as used by the manuscript figures.
MODEL_FAMILY: Dict[str, str] = {
    "lr": "LR",
    "lr_morph": "LR",
    "rf": "RF",
    "rf_morph": "RF",
    "cnn": "ResNet50",
    "cnn_morph": "ResNet50",
    "mlp_morph": "ResNet50",
}

#: Which models consume slide-level morphometric features.
USES_MORPHOMETRICS: Dict[str, bool] = {
    "lr": False,
    "lr_morph": True,
    "rf": False,
    "rf_morph": True,
    "cnn": False,
    "cnn_morph": True,
    "mlp_morph": True,
}

#: Models that ignore the image branch entirely (Table 2 ablation).
MORPHOMETRIC_ONLY: Dict[str, bool] = {
    "mlp_morph": True,
}

#: Deep-learning (torch) models vs. scikit-learn style models.
TORCH_MODELS = {"cnn", "cnn_morph", "mlp_morph"}
SKLEARN_MODELS = {"lr", "lr_morph", "rf", "rf_morph"}

#: Folder names used by the published ``evaluations_for_all`` directory, mapped
#: onto ``(task, model)`` pairs so reference results can be compared with
#: freshly produced ones.
#:
#: **Stored labels are folder labels.** Every ``predictions.csv`` in that
#: archive was written before the severity ordering existed, so its
#: ``true_label`` / ``predicted_label`` columns and its probability columns are
#: in the *folder* order. Read them through
#: :func:`vmlevaluator.evaluate.rescore.load_published_predictions`, which
#: applies the task's permutation; do not read them raw.
PUBLISHED_EVAL_FOLDERS: Dict[str, tuple] = {
    "LR_control_VML": ("control_vml", "lr"),
    "LR_numerical_control_vml": ("control_vml", "lr_morph"),
    "Resnet_control_vml": ("control_vml", "cnn"),
    "Resnet_numerical_control_vml": ("control_vml", "cnn_morph"),
    "LR_poor_very_poor": ("poor_very_poor", "lr"),
    "LR_numerical_poor_very_poor": ("poor_very_poor", "lr_morph"),
    "Resnet_poor_very_poor": ("poor_very_poor", "cnn"),
    "Resnet_numerical_poor_very_poor": ("poor_very_poor", "cnn_morph"),
    "LR_multiclass": ("multiclass", "lr"),
    "LR_multiclass_numerical": ("multiclass", "lr_morph"),
    "Resnet_multiclass": ("multiclass", "cnn"),
    "Resnet_multiclass_numerical": ("multiclass", "cnn_morph"),
}


#: Reverse map: ``(task, model)`` -> canonical experiment name used for outputs.
def experiment_name(task: str, model: str) -> str:
    return f"{model}__{task}"

"""VMLevaluator — reproducible pipeline for the VML muscle-histology study.

Integrating histology and morphometrics for deep-learning-based assessment of
skeletal muscle injury: patch extraction from laminin-stained whole-slide
images, three model families (logistic regression, XGBoost, ResNet-50), each
with and without slide-level morphometric fusion, evaluated at patch and
whole-slide level across three rat VML tasks, plus exploratory cross-species
transfer to human VL histology following ACL injury.

Quick start
-----------
::

    from vmlevaluator import load_experiment, train_experiment, evaluate_experiment

    cfg = load_experiment("multiclass", "cnn_morph", quick=True)
    train_experiment(cfg)
    result = evaluate_experiment(cfg)
    print(result.summary_row())

Data roots are set by environment variable — see :mod:`vmlevaluator.paths`.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import paths
from .config import describe, load_experiment, save_config
from .constants import MODELS, MODEL_LABELS, TASKS, TASK_LABELS

__all__ = [
    "__version__",
    "paths",
    "load_experiment",
    "describe",
    "save_config",
    "TASKS",
    "MODELS",
    "TASK_LABELS",
    "MODEL_LABELS",
    "train_experiment",
    "evaluate_experiment",
]


def __getattr__(name):
    """Import torch-dependent entry points lazily, so `import vmlevaluator`
    stays cheap and works even where torch is unavailable."""
    if name == "train_experiment":
        from .train import train_experiment

        return train_experiment
    if name == "evaluate_experiment":
        from .evaluate import evaluate_experiment

        return evaluate_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

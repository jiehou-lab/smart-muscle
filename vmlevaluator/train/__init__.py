"""Training entry points.

``train_experiment`` dispatches on the model key so notebooks and the CLI can
treat every one of the nine model variants the same way.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig

from ..constants import SKLEARN_MODELS, TORCH_MODELS
from .sklearn_trainer import SklearnTrainer, train_sklearn_model
from .torch_trainer import TorchTrainer, TrainingHistory, set_seed, train_torch_model

__all__ = [
    "train_experiment",
    "TorchTrainer",
    "SklearnTrainer",
    "TrainingHistory",
    "train_torch_model",
    "train_sklearn_model",
    "set_seed",
]


def train_experiment(cfg: DictConfig, *, verbose: bool = True) -> Dict[str, Any]:
    """Train whichever model ``cfg.meta.model`` names."""
    model = cfg.meta.model
    if model in TORCH_MODELS:
        return train_torch_model(cfg, verbose=verbose)
    if model in SKLEARN_MODELS:
        return train_sklearn_model(cfg, verbose=verbose)
    raise ValueError(f"No trainer registered for model {model!r}")

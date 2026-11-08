"""Build a model from a resolved config, and load saved checkpoints/bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from ..constants import TORCH_MODELS
from .networks import FusionClassifier, ImageClassifier, MorphometricClassifier

__all__ = [
    "build_model",
    "load_checkpoint",
    "save_sklearn_bundle",
    "load_sklearn_bundle",
    "select_device",
]


def select_device(cfg: Optional[DictConfig] = None) -> torch.device:
    """CUDA device from the config when available, otherwise CPU."""
    if torch.cuda.is_available():
        gpu_id = int(cfg.environment.gpu_id) if cfg is not None else 0
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def build_model(cfg: DictConfig, *, num_features: int = 2) -> torch.nn.Module:
    """Instantiate the torch model named by ``cfg.meta.model``."""
    model_key = cfg.meta.model
    if model_key not in TORCH_MODELS:
        raise ValueError(
            f"{model_key!r} is a scikit-learn model; build it through "
            "vmlevaluator.train.sklearn_models instead."
        )

    num_classes = int(cfg.data.num_classes)
    timm_cfg = cfg.models.timm
    num_cfg = cfg.models.numerical

    if model_key == "cnn":
        return ImageClassifier(
            model_name=str(timm_cfg.model_name),
            num_classes=num_classes,
            pretrained=bool(timm_cfg.pretrained),
        )
    if model_key == "cnn_morph":
        return FusionClassifier(
            model_name=str(timm_cfg.model_name),
            num_classes=num_classes,
            pretrained=bool(timm_cfg.pretrained),
            numerical_in_features=int(num_features),
            numerical_embed_dim=int(num_cfg.embed_dim),
            fusion_hidden_dim=int(num_cfg.fusion_hidden_dim),
            dropout_prob=float(num_cfg.dropout_prob),
        )
    if model_key == "mlp_morph":
        return MorphometricClassifier(
            num_classes=num_classes,
            numerical_in_features=int(num_features),
            numerical_embed_dim=int(num_cfg.embed_dim),
            hidden_dim=int(num_cfg.fusion_hidden_dim),
            dropout_prob=float(num_cfg.dropout_prob),
        )
    raise ValueError(f"Unknown torch model {model_key!r}")  # pragma: no cover


def load_checkpoint(
    model: torch.nn.Module, path, device: Optional[torch.device] = None, strict: bool = True
) -> torch.nn.Module:
    """Load a ``.pth`` state dict, tolerating legacy key prefixes.

    Checkpoints produced before this refactor already use matching keys; the
    ``module.`` prefix stripping covers weights saved from ``DataParallel``.
    """
    device = device or torch.device("cpu")
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint {path} does not match the model.\n"
            f"  missing keys   : {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )
    model.to(device)
    return model


# ── scikit-learn estimator bundles ───────────────────────────────────────────

def save_sklearn_bundle(
    path,
    model,
    scaler,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist estimator + fitted scaler + provenance as one joblib file."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "num_classes": int(num_classes),
            "class_names": list(class_names or []),
            "metadata": dict(metadata or {}),
        },
        p,
    )
    return p


def load_sklearn_bundle(path) -> Dict[str, Any]:
    import joblib

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    bundle = joblib.load(p)
    for key in ("model", "scaler", "num_classes"):
        if key not in bundle:
            raise KeyError(f"Invalid model bundle (missing {key!r}): {p}")
    return bundle

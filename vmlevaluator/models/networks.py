"""Torch models: image-only CNN, CNN+morphometric fusion, morphometric-only MLP.

Checkpoint compatibility
------------------------
Module attribute names are chosen so that ``state_dict`` keys match the
original code base exactly, and weights trained before this refactor load
without remapping:

============================  ==========================================
:class:`ImageClassifier`      ``model.model.*`` (timm), ``model.head.*``
:class:`FusionClassifier`     ``cnn.*``, ``num_encoder.net.*``, ``fusion_head.net.*``
:class:`MorphometricClassifier` ``num_encoder.net.*``, ``head.net.*``
============================  ==========================================

All three take a batch dict keyed by :data:`~vmlevaluator.constants.IMAGE` /
:data:`~vmlevaluator.constants.NUMERICAL` and return raw class logits, so
training, evaluation and inference share one call signature.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
from torch import nn

from ..constants import IMAGE, NUMERICAL

__all__ = [
    "ImageClassifier",
    "FusionClassifier",
    "MorphometricClassifier",
    "NumericalEncoder",
    "FusionHead",
]


def init_weights(module: nn.Module) -> None:
    """Xavier-uniform for linear layers, unit LayerNorm — as in the original."""
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _create_backbone(model_name: str, pretrained: bool):
    """Create a headless timm backbone.

    ImageNet weights are fetched from the model hub on first use. On a machine
    with no outbound network the download raises; rather than failing the whole
    run, fall back to random initialisation and say so loudly — the code path
    is then still exercisable offline (tests, CI, air-gapped review), while a
    warning makes clear that such a model is *not* comparable to the published
    one, which is ImageNet-pretrained.
    """
    import timm

    if not pretrained:
        return timm.create_model(model_name, pretrained=False, num_classes=0)

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=0)
    except Exception as exc:  # network, hub outage, cache miss offline
        warnings.warn(
            f"Could not download ImageNet weights for {model_name!r} ({type(exc).__name__}: "
            f"{exc}). Falling back to random initialisation. Results from this run are NOT "
            "comparable to the published models, which start from ImageNet weights. "
            "Restore network access, or pre-populate the timm cache, and re-run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return timm.create_model(model_name, pretrained=False, num_classes=0)


# ── image-only ───────────────────────────────────────────────────────────────

class _TimmBranch(nn.Module):
    """timm backbone + linear head. Named to match the legacy ``TIMM`` module."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = _create_backbone(model_name, pretrained)
        self.out_features = self.model.num_features
        self.head = nn.Linear(self.out_features, num_classes)
        self.head.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.model(x))


class ImageClassifier(nn.Module):
    """ResNet-50 (or any timm backbone) trained on histology patches alone.

    ``f_theta(x) -> 2048-d embedding -> linear head -> class logits``, i.e.
    Eq. (1) of the manuscript followed by the patch-level classifier.
    """

    def __init__(self, model_name: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = _TimmBranch(model_name, num_classes, pretrained)
        self.num_classes = num_classes
        self.embedding_dim = self.model.out_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch[IMAGE])

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features before the classification head."""
        return self.model.model(images)


# ── morphometric encoder + fusion head ───────────────────────────────────────

class NumericalEncoder(nn.Module):
    """Two-layer MLP mapping ``[avg CSA, myofiber count]`` to a 32-d embedding.

    Layer order — Linear, LayerNorm, GELU, Dropout, Linear, LayerNorm, GELU —
    is the morphometric branch described in the Methods.
    """

    def __init__(self, in_features: int = 2, embed_dim: int = 32, dropout_prob: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.out_features = embed_dim
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """Classification head over the concatenated representation."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout_prob: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionClassifier(nn.Module):
    """Intermediate fusion of CNN image features and slide-level morphometrics.

    ``z = [f_CNN(x) ; g_phi(m)]`` (Eq. 2), then the fusion head produces class
    logits. With ResNet-50 the concatenation is 2048 + 32 = 2080-d.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        numerical_in_features: int = 2,
        numerical_embed_dim: int = 32,
        fusion_hidden_dim: int = 256,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.cnn = _create_backbone(model_name, pretrained)
        cnn_dim = self.cnn.num_features

        self.num_encoder = NumericalEncoder(
            in_features=numerical_in_features,
            embed_dim=numerical_embed_dim,
            dropout_prob=dropout_prob,
        )
        self.fusion_head = FusionHead(
            in_features=cnn_dim + numerical_embed_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )
        self.num_classes = num_classes
        self.embedding_dim = cnn_dim + numerical_embed_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_feats = self.cnn(batch[IMAGE])
        num_feats = self.num_encoder(batch[NUMERICAL])
        return self.fusion_head(torch.cat([image_feats, num_feats], dim=1))

    @torch.no_grad()
    def embed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The fused 2080-d representation, before the head."""
        return torch.cat([self.cnn(batch[IMAGE]), self.num_encoder(batch[NUMERICAL])], dim=1)


class MorphometricClassifier(nn.Module):
    """Table-2 ablation: the fusion model's numerical branch, with no images.

    Same encoder and same head width as :class:`FusionClassifier`, so any
    difference in performance is attributable to the missing image branch
    rather than to capacity.
    """

    def __init__(
        self,
        num_classes: int = 2,
        numerical_in_features: int = 2,
        numerical_embed_dim: int = 32,
        hidden_dim: int = 256,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.num_encoder = NumericalEncoder(
            in_features=numerical_in_features,
            embed_dim=numerical_embed_dim,
            dropout_prob=dropout_prob,
        )
        self.head = FusionHead(
            in_features=numerical_embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )
        self.num_classes = num_classes

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[NUMERICAL] if isinstance(batch, dict) else batch
        return self.head(self.num_encoder(x))

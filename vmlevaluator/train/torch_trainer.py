"""Unified trainer for the three torch models.

One trainer replaces the four near-duplicate trainers of the original code
base (CNN binary, CNN multiclass, fusion binary, fusion multiclass). The
recipe is unchanged:

* AdamW (``lr``, ``weight_decay`` from config) with cosine annealing
  (``T_max=10``, ``eta_min=min_learning_rate``);
* class-weighted cross-entropy, weights = inverse frequency normalised to sum
  to ``num_classes``;
* per-epoch evaluation on the **selection split**, with the best-accuracy
  and best-macro-F1 checkpoints saved separately;
* early stopping once *both* best-epoch counters are more than ``patience``
  epochs behind.

The selection split is the validation split when ``data.validation`` is
configured (see :func:`vmlevaluator.data.loaders.validation_slides`); the test
split is then never looked at during training, and is scored once, afterwards,
by ``vmleval evaluate``. Without ``data.validation`` it is the test split, as in
the published runs -- which means the checkpoint is chosen on test, and the
test scores reported for it are optimistic.

Binary and multiclass share this path because both are trained with softmax
cross-entropy over ``num_classes`` logits, as described in the Methods.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from ..config import save_config
from ..constants import IMAGE, LABEL, NUMERICAL
from ..data.loaders import DataBundle, build_datasets, build_loaders, build_val_loader
from ..models.factory import build_model, select_device

__all__ = ["TrainingHistory", "TorchTrainer", "train_torch_model"]


def set_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    By default cuDNN autotuning stays enabled, matching the published runs, so
    GPU results are reproducible to within kernel-selection noise rather than
    bit-exact. ``deterministic=True`` (``environment.deterministic``) turns
    autotuning off and asks torch for deterministic kernels, so a re-run with
    the same seed, data and software reproduces the same weights -- at some
    cost in speed. Ops with no deterministic CUDA kernel warn instead of
    failing."""
    if deterministic:
        import os

        # Required by cuBLAS for deterministic matmuls; must precede CUDA init.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingHistory:
    """Per-epoch record, written to ``history.json`` and plotted in notebook 05."""

    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    test_macro_f1: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)
    #: "val" or "test": which split the best-* fields and checkpoints refer to.
    selection_split: str = "test"
    best_acc: float = 0.0
    best_macro_f1: float = 0.0
    best_acc_epoch: int = 0
    best_f1_epoch: int = 0
    seconds: float = 0.0

    def append(self, epoch, tr_loss, tr_acc, te_loss, te_acc, te_f1) -> None:
        """Record one epoch; the ``te_*`` values are on the selection split."""
        self.epoch.append(int(epoch))
        self.train_loss.append(float(tr_loss))
        self.train_acc.append(float(tr_acc))
        prefix = "val" if self.selection_split == "val" else "test"
        getattr(self, f"{prefix}_loss").append(float(te_loss))
        getattr(self, f"{prefix}_acc").append(float(te_acc))
        getattr(self, f"{prefix}_macro_f1").append(float(te_f1))

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "test_macro_f1": self.test_macro_f1,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_macro_f1": self.val_macro_f1,
            "selection_split": self.selection_split,
            "best_acc": self.best_acc,
            "best_macro_f1": self.best_macro_f1,
            "best_acc_epoch": self.best_acc_epoch,
            "best_f1_epoch": self.best_f1_epoch,
            "seconds": self.seconds,
        }

    def to_frame(self):
        import pandas as pd

        p = "val" if self.selection_split == "val" else "test"
        return pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_loss": self.train_loss,
                "train_acc": self.train_acc,
                f"{p}_loss": getattr(self, f"{p}_loss"),
                f"{p}_acc": getattr(self, f"{p}_acc"),
                f"{p}_macro_f1": getattr(self, f"{p}_macro_f1"),
            }
        )


class _AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def _build_scheduler(optimizer, cfg: DictConfig):
    name = str(cfg.optimization.scheduler).lower()
    if name in ("exponential", "epoential"):  # legacy configs carry the typo
        return lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.optimization.gamma))
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=float(cfg.optimization.min_learning_rate)
        )
    if name == "constant":
        return lr_scheduler.ConstantLR(optimizer)
    raise ValueError(f"Unknown scheduler {name!r}; expected cosine|exponential|constant.")


class TorchTrainer:
    """Trains ``cnn``, ``cnn_morph`` or ``mlp_morph`` from a resolved config."""

    def __init__(
        self,
        cfg: DictConfig,
        bundle: Optional[DataBundle] = None,
        *,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        self.cfg = cfg
        self.verbose = verbose
        set_seed(cfg.environment.seed,
                 deterministic=bool(cfg.environment.get("deterministic", False)))

        self.device = device or select_device(cfg)
        self.output_dir = Path(cfg.output.dir)
        (self.output_dir / "weights" / "acc").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "weights" / "f1").mkdir(parents=True, exist_ok=True)

        self.bundle = bundle or build_datasets(cfg, verbose=verbose)
        self.morphometric_only = self.bundle.morphometric_only
        self.uses_morphometrics = self.bundle.uses_morphometrics

        if self.bundle.morphometrics is not None and self.bundle.morphometrics.is_fitted:
            self.bundle.morphometrics.save_norm_stats(self.output_dir / "norm_stats.json")

        self.train_loader, self.test_loader = build_loaders(cfg, self.bundle)
        self.val_loader = build_val_loader(cfg, self.bundle)
        # Checkpoints and early stopping follow the validation split when there
        # is one; test is then not evaluated at all during training.
        self.select_loader = self.val_loader or self.test_loader
        self.select_name = "val" if self.val_loader is not None else "test"

        n_classes = int(cfg.data.num_classes)
        self.class_weights = self.bundle.train.class_weights(n_classes).to(self.device)

        n_features = (
            self.bundle.morphometrics.n_features if self.bundle.morphometrics is not None else 2
        )
        self.model = build_model(cfg, num_features=n_features).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg.optimization.learning_rate),
            weight_decay=float(cfg.optimization.weight_decay),
        )
        self.scheduler = _build_scheduler(self.optimizer, cfg)
        self.epochs = int(cfg.optimization.epochs)
        self.patience = int(cfg.optimization.patience)
        self.history = TrainingHistory(selection_split=self.select_name)

        save_config(cfg, self.output_dir / "config.yaml")
        self._log_file = open(self.output_dir / "logging.txt", "w", encoding="utf-8")

        if verbose:
            print(
                f"device={self.device}  model={cfg.meta.model}  task={cfg.meta.task}  "
                f"classes={n_classes}  epochs={self.epochs}"
            )
            print(f"class weights: {self.class_weights.cpu().tolist()}")
            print(f"checkpoint selection / early stopping on: {self.select_name} split")

    # ── internals ────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def _to_batch(self, items) -> Dict[str, torch.Tensor]:
        """Map a dataloader tuple onto the model's batch dict."""
        if self.morphometric_only:
            numerical, label = items[0], items[1]
            return {NUMERICAL: numerical.to(self.device), LABEL: label.to(self.device)}, label.to(
                self.device
            )
        if self.uses_morphometrics:
            image, numerical, label = items[0], items[1], items[2]
            label = label.to(self.device)
            return (
                {
                    IMAGE: image.to(self.device),
                    NUMERICAL: numerical.to(self.device),
                    LABEL: label,
                },
                label,
            )
        image, label = items[0], items[1]
        label = label.to(self.device)
        return {IMAGE: image.to(self.device), LABEL: label}, label

    def _run_epoch(self, loader: DataLoader, train: bool):
        loss_meter, acc_meter = _AverageMeter(), _AverageMeter()
        y_true: List[int] = []
        y_pred: List[int] = []

        self.model.train(train)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for items in loader:
                batch, label = self._to_batch(items)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = F.cross_entropy(logits, label, weight=self.class_weights)
                if train:
                    loss.backward()
                    self.optimizer.step()

                preds = logits.argmax(dim=1)
                correct = (preds == label).float().sum().item()
                loss_meter.update(loss.item(), logits.size(0))
                acc_meter.update(100.0 * correct / logits.size(0), logits.size(0))
                y_true.extend(label.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return loss_meter.avg, acc_meter.avg, macro_f1

    # ── public API ───────────────────────────────────────────────────────────

    def train(self) -> TrainingHistory:
        best_acc = best_f1 = 0.0
        best_acc_epoch = best_f1_epoch = 0
        start = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, _ = self._run_epoch(self.train_loader, train=True)
            self.scheduler.step()
            test_loss, test_acc, test_f1 = self._run_epoch(self.select_loader, train=False)
            self.history.append(epoch, train_loss, train_acc, test_loss, test_acc, test_f1)

            if test_acc > best_acc:
                best_acc, best_acc_epoch = test_acc, epoch
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "weights" / "acc" / "best_model.pth",
                )
            if test_f1 > best_f1:
                best_f1, best_f1_epoch = test_f1, epoch
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "weights" / "f1" / "best_model.pth",
                )

            self._log(
                f"epoch {epoch:03d} | train loss {train_loss:.5f} acc {train_acc:6.3f}% | "
                f"{self.select_name} loss {test_loss:.5f} acc {test_acc:6.3f}% "
                f"macro-F1 {test_f1:.4f} | "
                f"best acc {best_acc:.3f}%@{best_acc_epoch} best F1 {best_f1:.4f}@{best_f1_epoch}"
            )

            if (epoch - best_acc_epoch) > self.patience and (epoch - best_f1_epoch) > self.patience:
                self._log(f"early stopping at epoch {epoch} (patience={self.patience})")
                break

        self.history.best_acc = best_acc
        self.history.best_macro_f1 = best_f1
        self.history.best_acc_epoch = best_acc_epoch
        self.history.best_f1_epoch = best_f1_epoch
        self.history.seconds = time.time() - start

        (self.output_dir / "history.json").write_text(
            json.dumps(self.history.to_dict(), indent=2), encoding="utf-8"
        )
        self._log(
            f"done in {self.history.seconds:.1f}s — best {self.select_name} acc {best_acc:.3f}%, "
            f"best macro-F1 {best_f1:.4f}"
        )
        self._log_file.close()
        return self.history

    @property
    def best_checkpoint(self) -> Path:
        """Best-macro-F1 weights — the checkpoint used for reported metrics."""
        return self.output_dir / "weights" / "f1" / "best_model.pth"


def train_torch_model(cfg: DictConfig, *, verbose: bool = True) -> Dict[str, object]:
    """Convenience wrapper: train and return paths plus history."""
    trainer = TorchTrainer(cfg, verbose=verbose)
    history = trainer.train()
    return {
        "output_dir": str(trainer.output_dir),
        "checkpoint": str(trainer.best_checkpoint),
        "history": history,
        "data": trainer.bundle.summary(),
    }

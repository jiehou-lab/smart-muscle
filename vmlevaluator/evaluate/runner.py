"""Run a trained model over a split and write the full evaluation bundle.

``evaluate_experiment`` is model-agnostic: it finds the right predictor
(torch checkpoint or joblib bundle), collects patch-level probabilities,
aggregates them to slides, computes both metric levels and writes every
artefact described in :mod:`vmlevaluator.evaluate.report`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..constants import IMAGE, MORPHOMETRIC_ONLY, NUMERICAL, TORCH_MODELS
from ..data.datasets import MorphometricTableDataset, PatchDataset
from ..data.loaders import build_datasets, build_morphometrics
from ..features import extract_features, feature_bytes, iter_feature_chunks
from ..models.factory import build_model, load_checkpoint, load_sklearn_bundle, select_device
from . import report as report_io
from .aggregate import aggregate_to_wsi, wsi_metrics
from .metrics import MetricSet, compute_metrics, predict_labels

__all__ = ["EvaluationResult", "predict_patches", "evaluate_experiment", "evaluate_predictions_csv"]


@dataclass
class EvaluationResult:
    """Everything one evaluation produces, ready for notebook display."""

    task: str
    model: str
    task_type: str
    output_dir: Path
    patch: MetricSet
    wsi: MetricSet
    predictions: pd.DataFrame
    wsi_predictions: pd.DataFrame

    def summary_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "patch_accuracy": self.patch.accuracy,
            "patch_f1": self.patch.headline_f1,
            "patch_macro_f1": self.patch.macro_f1,
            "patch_auc": self.patch.auc if self.task_type == "binary" else self.patch.macro_auc,
            "wsi_accuracy": self.wsi.accuracy,
            "wsi_f1": self.wsi.headline_f1,
            "wsi_macro_f1": self.wsi.macro_f1,
            "n_patches": self.patch.n,
            "n_wsi": self.wsi.n,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<EvaluationResult {self.model} / {self.task}: "
            f"patch F1 {self.patch.headline_f1:.3f}, WSI F1 {self.wsi.headline_f1:.3f}>"
        )


def _task_type(cfg: DictConfig) -> str:
    return "binary" if int(cfg.data.num_classes) == 2 else "multiclass"


def _find_checkpoint(cfg: DictConfig, model_path: Optional[str], torch_model: bool) -> Path:
    if model_path is not None:
        return Path(model_path)
    base = Path(cfg.output.dir)
    candidate = (
        base / "weights" / "f1" / "best_model.pth"
        if torch_model
        else base / "weights" / "acc" / "best_model.joblib"
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"No trained model found at {candidate}. Train it first, or pass model_path=..."
        )
    return candidate


# ── prediction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_torch(
    cfg: DictConfig, dataset, checkpoint: Path, device: torch.device, num_features: int
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    model = build_model(cfg, num_features=num_features)
    load_checkpoint(model, checkpoint, device=device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )
    morph_only = MORPHOMETRIC_ONLY.get(cfg.meta.model, False)
    uses_morph = bool(getattr(dataset, "with_morphometrics", False)) or morph_only

    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    paths: List[str] = []

    for items in loader:
        if morph_only:
            numerical, label, path = items[0], items[1], items[2]
            batch = {NUMERICAL: numerical.to(device)}
        elif uses_morph:
            image, numerical, label, path = items[0], items[1], items[2], items[3]
            batch = {IMAGE: image.to(device), NUMERICAL: numerical.to(device)}
        else:
            image, label, path = items[0], items[1], items[2]
            batch = {IMAGE: image.to(device)}

        logits = model(batch)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(label.numpy())
        paths.extend(list(path))

    return np.concatenate(probs), paths, np.concatenate(labels)


def _predict_sklearn(
    cfg: DictConfig, dataset, bundle_path: Path
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    bundle = load_sklearn_bundle(bundle_path)

    if MORPHOMETRIC_ONLY.get(cfg.meta.model, False):
        X, y, paths = dataset.to_arrays()
    else:
        threads = int(cfg.data.get("loader_threads", 0) or 0)
        budget = float(cfg.get("streaming", {}).get("max_in_memory_gb", 1e9))
        need = feature_bytes(len(dataset), morphometrics=bool(
            getattr(dataset, "with_morphometrics", False))) / 2**30
        if need > budget:
            # Too large to hold at once: score chunk by chunk, same order.
            probs, ys, ps = [], [], []
            for Xc, yc, pc in iter_feature_chunks(
                    dataset, int(cfg.streaming.chunk_patches),
                    batch_size=int(cfg.data.batch_size), threads=threads):
                probs.append(bundle["model"].predict_proba(
                    bundle["scaler"].transform(Xc).astype(np.float32, copy=False)))
                ys.append(yc)
                ps.extend(pc)
            return np.concatenate(probs), ps, np.concatenate(ys)
        X, y, paths = extract_features(
            dataset,
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            return_paths=True,
            verbose=False,
            threads=threads,
        )

    X_scaled = bundle["scaler"].transform(X)
    probs = bundle["model"].predict_proba(X_scaled)
    return np.asarray(probs), list(paths), np.asarray(y)


def predict_patches(
    cfg: DictConfig,
    *,
    dataset=None,
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Patch-level class probabilities for the configured test split.

    Returns ``(probabilities, patch_paths, true_labels)``.
    """
    is_torch = cfg.meta.model in TORCH_MODELS
    checkpoint = _find_checkpoint(cfg, model_path, is_torch)

    if dataset is None:
        norm_stats = Path(cfg.output.dir) / "norm_stats.json"
        lookup = None
        if cfg.uses_morphometrics:
            lookup = build_morphometrics(
                cfg,
                norm_stats_path=norm_stats if norm_stats.is_file() else None,
                verbose=verbose,
            )
        bundle = build_datasets(
            cfg,
            morphometrics=lookup,
            train_augmentation=False,
            return_paths_test=True,
            with_validation=False,   # only the test split is scored here
            verbose=verbose,
        )
        dataset = bundle.test
        num_features = bundle.morphometrics.n_features if bundle.morphometrics else 2
    else:
        num_features = 2

    if verbose:
        print(f"evaluating {cfg.meta.model} / {cfg.meta.task} from {checkpoint}")

    if is_torch:
        return _predict_torch(cfg, dataset, checkpoint, device or select_device(cfg), num_features)
    return _predict_sklearn(cfg, dataset, checkpoint)


# ── full evaluation ──────────────────────────────────────────────────────────

def evaluate_experiment(
    cfg: DictConfig,
    *,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    make_plots: bool = True,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate a trained experiment and write the full artefact bundle."""
    task_type = _task_type(cfg)
    class_names = list(cfg.data.class_names)
    out_dir = Path(output_dir or (Path(cfg.output.dir) / "evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    y_proba, paths, y_true = predict_patches(
        cfg, model_path=model_path, verbose=verbose
    )
    y_pred = predict_labels(y_proba, task_type, threshold)

    patch = compute_metrics(
        y_true, y_pred, y_proba, task_type=task_type, class_names=class_names, level="patch"
    )
    report_io.write_predictions(out_dir, paths, y_true, y_pred, y_proba)
    report_io.write_patch_report(out_dir, patch, y_true, y_pred, y_proba, class_names)

    predictions = pd.read_csv(out_dir / "predictions.csv")
    wsi_table = aggregate_to_wsi(predictions, task_type=task_type, threshold=threshold)
    wsi = wsi_metrics(wsi_table, task_type=task_type, class_names=class_names)
    report_io.write_wsi_report(out_dir, wsi_table, wsi, class_names, threshold=threshold)

    if make_plots:
        report_io.plot_confusion_matrix(out_dir, y_true, y_pred, class_names)
        report_io.plot_roc(out_dir, y_true, y_proba, class_names, task_type)

    report_io.write_metrics_json(
        out_dir,
        {
            "task": cfg.meta.task,
            "model": cfg.meta.model,
            "task_type": task_type,
            "class_names": class_names,
            "threshold": threshold,
            "quick_run": bool(cfg.quick_run.enabled),
            "patch": patch.to_dict(),
            "wsi": wsi.to_dict(),
        },
    )

    if verbose:
        print(
            f"  patch: accuracy {patch.accuracy:.4f}, F1 {patch.headline_f1:.4f}\n"
            f"  WSI  : accuracy {wsi.accuracy:.4f}, F1 {wsi.headline_f1:.4f}  "
            f"({wsi.n} slides)\n  artefacts -> {out_dir}"
        )

    return EvaluationResult(
        task=cfg.meta.task,
        model=cfg.meta.model,
        task_type=task_type,
        output_dir=out_dir,
        patch=patch,
        wsi=wsi,
        predictions=predictions,
        wsi_predictions=wsi_table,
    )


def evaluate_predictions_csv(
    predictions_csv,
    *,
    task_type: str = "binary",
    class_names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, MetricSet]:
    """Recompute patch- and WSI-level metrics from an existing predictions CSV.

    Thin alias for :func:`vmlevaluator.evaluate.rescore.rescore_predictions`,
    which carries the implementation and imports no torch — archived results,
    including the published ``evaluations_for_all/`` folders, can be re-scored
    on a machine with no deep learning stack installed.
    """
    from .rescore import rescore_predictions

    return rescore_predictions(
        predictions_csv,
        task_type=task_type,
        class_names=class_names,
        threshold=threshold,
    )

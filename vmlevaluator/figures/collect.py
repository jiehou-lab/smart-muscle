"""Collect evaluation results into one tidy table.

Every manuscript figure is a view of the same table: one row per
``(task, model)``, carrying patch- and WSI-level metrics. Metrics are
**recomputed from each run's ``predictions.csv``** rather than scraped from
summary text, so a figure can never drift away from the predictions that
produced it.

Two directory layouts are understood:

``results root`` (fresh runs)
    ``<results>/<model>__<task>/evaluation/predictions.csv``
``archive`` (the published ``evaluations_for_all/``)
    ``<archive>/<Resnet_multiclass_numerical>/predictions.csv``, mapped onto
    ``(task, model)`` by :data:`~vmlevaluator.constants.PUBLISHED_EVAL_FOLDERS`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..constants import (
    MODEL_FAMILY,
    MODEL_LABELS,
    PUBLISHED_EVAL_FOLDERS,
    TASK_CLASS_NAMES,
    TASK_LABELS,
    TASK_NUM_CLASSES,
    USES_MORPHOMETRICS,
)
from ..evaluate.rescore import rescore_predictions

__all__ = [
    "collect_results",
    "collect_from_results_root",
    "collect_from_archive",
    "load_summary",
    "SUMMARY_COLUMNS",
]

SUMMARY_COLUMNS = [
    "task", "task_label", "task_type", "model", "model_label", "family", "morphometrics",
    "n_patches", "patch_accuracy", "patch_f1", "patch_macro_f1", "patch_auc",
    # One-vs-rest for the most injured class -- identical to patch_f1 on a
    # binary task (classes are severity ordered), and the answer to "how well is
    # Very poor found?" on the three-class task, which macro-F1 averages away.
    "patch_most_injured_f1", "patch_most_injured_recall", "patch_most_injured_auc",
    "most_injured_class",
    "n_wsi", "wsi_accuracy", "wsi_f1", "wsi_macro_f1", "wsi_most_injured_f1", "source",
]


def _row_from_predictions(task: str, model: str, predictions_csv: Path) -> Dict:
    task_type = "binary" if TASK_NUM_CLASSES[task] == 2 else "multiclass"
    scored = rescore_predictions(
        predictions_csv, task_type=task_type, class_names=TASK_CLASS_NAMES[task]
    )
    patch, wsi = scored["patch"], scored["wsi"]
    return {
        "task": task,
        "task_label": TASK_LABELS[task],
        "task_type": task_type,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "family": MODEL_FAMILY[model],
        "morphometrics": USES_MORPHOMETRICS[model],
        "n_patches": patch.n,
        "patch_accuracy": patch.accuracy,
        "patch_f1": patch.headline_f1,
        "patch_macro_f1": patch.macro_f1,
        "patch_auc": patch.auc if task_type == "binary" else patch.macro_auc,
        "patch_most_injured_f1": patch.most_injured_f1,
        "patch_most_injured_recall": patch.most_injured_recall,
        "patch_most_injured_auc": patch.most_injured_auc,
        "most_injured_class": patch.most_injured_class,
        "n_wsi": wsi.n,
        "wsi_accuracy": wsi.accuracy,
        "wsi_f1": wsi.headline_f1,
        "wsi_macro_f1": wsi.macro_f1,
        "wsi_most_injured_f1": wsi.most_injured_f1,
        "source": str(predictions_csv),
    }


def collect_from_results_root(results_root, verbose: bool = True) -> pd.DataFrame:
    """Scan ``<results>/<model>__<task>/`` for per-patch predictions.

    Two layouts are accepted: ``vmleval evaluate`` puts its output in an
    ``evaluation/`` subfolder, while older result folders keep ``predictions.csv``
    directly in the experiment folder. Both are tried, so neither comes back
    silently empty.
    """
    root = Path(results_root)
    rows: List[Dict] = []
    if not root.is_dir():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = re.fullmatch(r"(?P<model>[a-z_]+)__(?P<task>[a-z_]+)", exp_dir.name)
        if not match:
            continue
        model, task = match.group("model"), match.group("task")
        if task not in TASK_NUM_CLASSES or model not in MODEL_LABELS:
            continue
        for predictions in (exp_dir / "evaluation" / "predictions.csv",
                            exp_dir / "predictions.csv"):
            if predictions.is_file():
                rows.append(_row_from_predictions(task, model, predictions))
                break
        else:
            if verbose:
                print(f"  skipping {exp_dir.name}: no predictions.csv "
                      "(looked in the folder and in evaluation/)")

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def collect_from_archive(archive_root, verbose: bool = True) -> pd.DataFrame:
    """Scan a published ``evaluations_for_all/`` directory."""
    root = Path(archive_root)
    rows: List[Dict] = []
    if not root.is_dir():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for folder, (task, model) in PUBLISHED_EVAL_FOLDERS.items():
        predictions = root / folder / "predictions.csv"
        if not predictions.is_file():
            if verbose:
                print(f"  missing: {folder}/predictions.csv")
            continue
        rows.append(_row_from_predictions(task, model, predictions))

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def collect_results(
    results_root=None,
    archive_root=None,
    *,
    prefer: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge fresh and archived results into one table.

    ``prefer`` decides which source wins when both contain the same
    ``(task, model)``: ``"results"`` (a run you just produced) or
    ``"archive"`` (the published numbers).
    """
    fresh = collect_from_results_root(results_root, verbose) if results_root else pd.DataFrame(
        columns=SUMMARY_COLUMNS
    )
    archived = collect_from_archive(archive_root, verbose) if archive_root else pd.DataFrame(
        columns=SUMMARY_COLUMNS
    )

    if fresh.empty:
        combined = archived
    elif archived.empty:
        combined = fresh
    else:
        first, second = (fresh, archived) if prefer == "results" else (archived, fresh)
        combined = pd.concat([first, second], ignore_index=True)
        combined = combined.drop_duplicates(subset=["task", "model"], keep="first")

    if combined.empty:
        return combined

    task_order = {t: i for i, t in enumerate(TASK_NUM_CLASSES)}
    family_order = {"LR": 0, "RF": 1, "ResNet50": 2}
    combined = combined.assign(
        _t=combined["task"].map(task_order),
        _f=combined["family"].map(family_order),
        _m=combined["morphometrics"].astype(int),
    ).sort_values(["_t", "_f", "_m"]).drop(columns=["_t", "_f", "_m"])
    return combined.reset_index(drop=True)


def load_summary(path) -> pd.DataFrame:
    """Read a previously saved ``master_summary.csv``."""
    return pd.read_csv(path)


def save_summary(df: pd.DataFrame, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p

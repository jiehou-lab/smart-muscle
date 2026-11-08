"""Write evaluation artefacts in the layout the figure pipeline expects.

Each evaluation directory ends up holding::

    predictions.csv             one row per patch: labels + class probabilities
    prediction_patch_level.csv  compact name/label/logits/prob form
    wsi_predictions.csv         one row per slide after probability averaging
    classification_report.txt   headline metrics + sklearn table
    detailed_metrics.txt        per-class precision / recall / F1 / AUC
    wsi_level_results.txt       slide-level metrics and per-slide table
    metrics.json                everything above, machine-readable
    confusion_matrix.png, roc_curve.png

The text files keep the exact headings the published runs used
("Overall Accuracy:", "WSI-Level Macro F1:", ...) so that
:mod:`vmlevaluator.figures.collect` can parse new and archived results with
the same code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import MetricSet, confusion, ovr_roc, text_report

__all__ = [
    "write_predictions",
    "write_patch_report",
    "write_wsi_report",
    "write_metrics_json",
    "plot_confusion_matrix",
    "plot_roc",
]


def write_predictions(
    output_dir,
    paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Path]:
    """Write ``predictions.csv`` and ``prediction_patch_level.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = y_proba.shape[1]

    df = pd.DataFrame(
        {
            "filename": [Path(p).name for p in paths],
            "filepath": list(paths),
            "true_label": np.asarray(y_true, dtype=int),
            "predicted_label": np.asarray(y_pred, dtype=int),
        }
    )
    for i in range(num_classes):
        df[f"predicted_prob_class_{i}"] = y_proba[:, i]
    predictions_csv = output_dir / "predictions.csv"
    df.to_csv(predictions_csv, index=False)

    # Compact legacy form; log-probabilities stand in for logits so that the
    # file is self-consistent for both torch and scikit-learn models.
    logits = np.log(np.clip(y_proba, 1e-10, None))
    patch_csv = output_dir / "prediction_patch_level.csv"
    with open(patch_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["name", "label"]
            + [f"logits_{i}" for i in range(num_classes)]
            + [f"prob_{i}" for i in range(num_classes)]
        )
        for idx, path in enumerate(paths):
            writer.writerow(
                [Path(path).name, int(y_true[idx])]
                + [float(v) for v in logits[idx]]
                + [float(v) for v in y_proba[idx]]
            )

    return {"predictions": predictions_csv, "patch_level": patch_csv}


def write_patch_report(
    output_dir,
    metrics: MetricSet,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Path]:
    """``classification_report.txt`` + ``detailed_metrics.txt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    binary = metrics.task_type == "binary"

    lines: List[str] = ["Classification Report", "=" * 50, ""]
    lines.append(f"Overall Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    lines.append("")
    if binary:
        if metrics.auc is not None:
            lines += [f"ROC AUC: {metrics.auc:.4f}", ""]
    else:
        lines.append(f"Macro Precision: {metrics.macro_precision:.4f}")
        lines.append(f"Macro Recall: {metrics.macro_recall:.4f}")
        lines.append(f"Macro F1: {metrics.macro_f1:.4f}")
        if metrics.macro_auc is not None:
            lines.append(f"Macro AUC (OvR): {metrics.macro_auc:.4f}")
        lines.append("")
    lines.append(text_report(y_true, y_pred, class_names))
    report_path = output_dir / "classification_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    curves, macro_auc, weighted_auc = ovr_roc(
        np.asarray(y_true), np.asarray(y_proba), num_classes, class_names
    )
    detail: List[str] = ["Detailed Metrics Summary", "=" * 50, ""]
    detail.append(f"Overall Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    detail.append(f"Macro Precision: {metrics.macro_precision:.4f}")
    detail.append(f"Macro Recall: {metrics.macro_recall:.4f}")
    detail.append(f"Macro F1: {metrics.macro_f1:.4f}")
    if binary and metrics.auc is not None:
        detail.append(f"ROC AUC: {metrics.auc:.4f}")
        detail.append(f"Positive class: {metrics.pos_class}")
        detail.append(f"Positive-class Precision: {metrics.pos_precision:.4f}")
        detail.append(f"Positive-class Recall: {metrics.pos_recall:.4f}")
        detail.append(f"Positive-class F1: {metrics.pos_f1:.4f}")
    if macro_auc is not None:
        detail.append(f"Macro AUC (OvR): {macro_auc:.4f}")
    if weighted_auc is not None:
        detail.append(f"Weighted AUC (OvR): {weighted_auc:.4f}")
    detail += ["", "Per-Class AUC (OvR):", "-" * 50]
    for idx, curve in curves.items():
        value = f"{curve['auc']:.4f}" if curve["auc"] is not None else "N/A (single class present)"
        detail.append(f"  {curve['name']}: {value}")
    detail_path = output_dir / "detailed_metrics.txt"
    detail_path.write_text("\n".join(detail), encoding="utf-8")

    return {"classification_report": report_path, "detailed_metrics": detail_path}


def write_wsi_report(
    output_dir,
    wsi_table: pd.DataFrame,
    metrics: MetricSet,
    class_names: Sequence[str],
    *,
    threshold: float = 0.5,
) -> Dict[str, Path]:
    """``wsi_level_results.txt`` + ``wsi_predictions.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = metrics.task_type == "binary"
    prob_cols = sorted(
        [c for c in wsi_table.columns if c.startswith("predicted_prob_class_")],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )

    csv_path = output_dir / "wsi_predictions.csv"
    wsi_table.to_csv(csv_path, index=False)

    lines: List[str] = ["=" * 60, "WSI-Level Evaluation Results", "=" * 60, ""]
    lines.append(f"Total WSIs: {len(wsi_table)}")
    if binary:
        lines.append(f"Decision Threshold: {threshold:.4f}")
        lines.append(f"WSI-Level Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
        if metrics.auc is not None:
            lines.append(f"WSI-Level AUC: {metrics.auc:.4f}")
        lines.append(f"WSI-Level F1-Score: {metrics.pos_f1:.4f}")
        lines.append(f"WSI-Level Precision: {metrics.pos_precision:.4f}")
        lines.append(f"WSI-Level Recall: {metrics.pos_recall:.4f}")
    else:
        lines.append(f"WSI-Level Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
        lines.append(f"WSI-Level Macro Precision: {metrics.macro_precision:.4f}")
        lines.append(f"WSI-Level Macro Recall: {metrics.macro_recall:.4f}")
        lines.append(f"WSI-Level Macro F1: {metrics.macro_f1:.4f}")

    lines += ["", "Per-Class Statistics:", "-" * 60]
    y = wsi_table["true_label"].to_numpy()
    pred = wsi_table["predicted_label"].to_numpy()
    for label in sorted(np.unique(y)):
        mask = y == label
        name = class_names[label] if label < len(class_names) else str(label)
        lines.append(
            f"{name} (label {label}): {int(mask.sum())} WSIs, "
            f"Accuracy: {float(np.mean(pred[mask] == label)):.4f}"
        )

    lines += ["", "WSI-Level Predictions:", "-" * 60]
    header = f"{'WSI ID':<34} {'True':>6} {'Pred':>6} {'N':>6}"
    for i in range(len(prob_cols)):
        header += f" {'mean_p' + str(i):>10}"
    lines += [header, "-" * 60]
    for _, row in wsi_table.iterrows():
        line = (
            f"{str(row['wsi_id'])[:34]:<34} {int(row['true_label']):>6} "
            f"{int(row['predicted_label']):>6} {int(row.get('n_patches', 0)):>6}"
        )
        for col in prob_cols:
            line += f" {float(row[col]):>10.4f}"
        lines.append(line)

    txt_path = output_dir / "wsi_level_results.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return {"wsi_results": txt_path, "wsi_predictions": csv_path}


def write_metrics_json(output_dir, payload: Dict) -> Path:
    path = Path(output_dir) / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


# ── plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(output_dir, y_true, y_pred, class_names: Sequence[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    cm = confusion(y_true, y_pred, len(class_names))
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 3, 1.4 * len(class_names) + 2.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix", fontweight="bold")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:d}", ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc(output_dir, y_true, y_proba, class_names: Sequence[str], task_type: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if task_type == "binary":
        from .metrics import binary_roc

        roc = binary_roc(y_true, y_proba[:, 1])
        if roc is not None:
            fpr, tpr, auc_value = roc
            ax.plot(fpr, tpr, lw=2.4, label=f"{class_names[1]} (AUC = {auc_value:.3f})")
        title = "ROC curve"
    else:
        curves, macro_auc, _ = ovr_roc(y_true, y_proba, len(class_names), class_names)
        for curve in curves.values():
            if curve["fpr"] is None:
                continue
            ax.plot(curve["fpr"], curve["tpr"], lw=2.2,
                    label=f"{curve['name']} (AUC = {curve['auc']:.3f})")
        title = "ROC curve (one-vs-rest)"
        if macro_auc is not None:
            title += f"\nmacro AUC = {macro_auc:.4f}"

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(output_dir) / "roc_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

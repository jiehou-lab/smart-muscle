"""Regenerate every manuscript figure and table from evaluation outputs.

=========  ===========================================================
Figure 1   Patch-level F1 across the three rat tasks (image-only vs fusion)
Figure 2   ROC curves per task (binary ROC; one-vs-rest macro for 3-class)
Figure 3   WSI-level F1 across the three rat tasks
Figure 4   Cross-species transfer to human VL histology following ACL injury
Table 1    Dataset sizes and task-specific train/test splits
Table 2    ResNet-50 ablation: image-only vs morphometric-only vs fusion
=========  ===========================================================

Figures 1-3 are driven by the tidy table from
:mod:`vmlevaluator.figures.collect`, which recomputes metrics from each run's
``predictions.csv``. Figure 4 reads the recorded transfer metrics, because the
human VL images are not redistributable — see
``results_reference/aclr_transfer_metrics.csv``.

Binary panels report positive-class F1; the three-class panel reports
macro-averaged F1, exactly as the captions state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..constants import (MODEL_FAMILY, MODELS, MORPHOMETRIC_ONLY, TASK_CLASS_NAMES,
                          TASK_LABELS, TASK_NUM_CLASSES, USES_MORPHOMETRICS)
from ..evaluate.metrics import binary_roc, ovr_roc
from . import style

__all__ = [
    "figure1_patch_f1",
    "figure2_roc",
    "figure3_wsi_f1",
    "figure4_aclr_transfer",
    "table1_datasets",
    "table2_ablation",
    "build_all",
]

def _image_only_models() -> List[str]:
    """The image-only member of each model family, in registry order.

    Derived rather than written down, so this module stays correct in a package
    whose middle family is a random forest and in one whose middle family is
    gradient boosting. A model qualifies if it takes the image, is not one of
    the morphometric-only ablations, and is not a constrained variant — that is
    exactly the set the manuscript figures pair with their fused counterparts.
    """
    return [m for m in MODELS
            if not MORPHOMETRIC_ONLY.get(m, False)
            and not USES_MORPHOMETRICS.get(m, False)
            and not m.endswith("_mono")]


#: The families the panels show, left to right.
FAMILY_ORDER = list(dict.fromkeys(MODEL_FAMILY[m] for m in _image_only_models()))
TASK_ORDER = ["control_vml", "poor_very_poor", "multiclass"]

#: Panel headings, as they appear in the manuscript. The second line names what
#: the panel is actually scoring: the positive class for a binary task, the
#: three grades for the multiclass one.
PANEL_TITLES = {
    "control_vml": "Control vs VML\n(F1 for VML)",
    "poor_very_poor": "Poor vs Very poor\n(F1 for Very poor)",
    "multiclass": "Multiclass\n(Perfect / Poor / Very poor)",
}

#: Short headings for the ROC panels, which have no room for a second line.
ROC_PANEL_TITLES = {
    "control_vml": "Control vs VML",
    "poor_very_poor": "Poor vs Very poor",
    "multiclass": "Multiclass",
}

#: How a fused model is named in a legend — the morphometrics are the two
#: numerical slide-level features, hence "+Num".
FUSION_SUFFIX = "+Num"

#: How a family is spelled on an axis. The registry uses compact keys; a figure
#: has room for the real name.
FAMILY_DISPLAY = {"ResNet50": "ResNet-50", "XGB": "XGBoost"}

#: Whether Figure 3's panel titles carry the test-slide count. On by default
#: because slide-level scores over 6-11 slides move in visible steps and a
#: reader should see the denominator; set False to reproduce a figure set that
#: was published without it.
SHOW_WSI_COUNT_IN_TITLE = True


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    style.apply_rcparams()
    return plt


def _save(fig, out_dir, name: str, caption: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    (out_dir / f"{name}_caption.txt").write_text(caption, encoding="utf-8")
    return png


#: The six models the manuscript figures show: one image-only and one fused
#: variant per family. Ablations (morphometric-only, monotone-constrained) are
#: real results and live in Table 2 and Table 3, but putting them in a
#: three-bar-pair panel would make the figure unreadable and would compare
#: things the figure is not about.
FIGURE_MODELS = [m for base in _image_only_models()
                 for m in (base, f"{base}_morph") if m in MODELS]


def _panel_frame(summary: pd.DataFrame, task: str,
                 models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Rows for one task, ordered LR, LR+morph, RF, RF+morph, ResNet, ResNet+morph."""
    keep = list(models or FIGURE_MODELS)
    sub = summary[(summary["task"] == task) & (summary["model"].isin(keep))].copy()
    sub["_f"] = sub["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    sub["_m"] = sub["morphometrics"].astype(int)
    return sub.sort_values(["_f", "_m"]).drop(columns=["_f", "_m"])


def _metric_name(task: str) -> str:
    """Which F1 a panel reports, for captions and tables.

    Binary panels name the positive class rather than saying "positive-class",
    because with severity-ordered labels that class is always the more injured
    one and a reader should be able to see which without the caption.
    """
    if TASK_NUM_CLASSES[task] > 2:
        return "macro F1"
    return f"F1 for {TASK_CLASS_NAMES[task][1]}"


#: Legend wording for the two bars in every family group.
BAR_LABELS = {"image": "Image only", "fusion": "Fusion (image + morphometrics)"}


def _grouped_bars(ax, sub: pd.DataFrame, value_column: str, ylabel: str, title: str,
                  figure_key: str = "patch") -> None:
    """One pair of bars per family: image-only and fused.

    Colour separates the two *inputs*, not the three families — the family is
    already written under its group on the x-axis, so a family-coloured bar
    would say nothing the axis does not. Hatching repeats the same distinction
    so the panel reads correctly in grayscale.
    """
    width = 0.36
    positions = np.arange(len(FAMILY_ORDER), dtype=float)

    # A panel with nothing in it means that task was not part of this run --
    # say so, rather than leaving a reader to wonder whether every model scored
    # zero. This is the normal case when someone runs a single model.
    if sub.empty:
        ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center",
                fontsize=style.TYPOGRAPHY["label"], color="#9e9e9e", style="italic")

    for offset, morph in ((-width / 2, False), (width / 2, True)):
        values, xs = [], []
        for i, family in enumerate(FAMILY_ORDER):
            row = sub[(sub["family"] == family) & (sub["morphometrics"] == morph)]
            if row.empty:
                continue
            values.append(float(row.iloc[0][value_column]))
            xs.append(positions[i] + offset)
        if not values:
            continue
        bars = ax.bar(
            xs,
            values,
            width=width,
            color=style.fusion_color(figure_key, morph),
            hatch=style.fusion_hatch(morph),
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        style.label_bars(ax, bars, values)

    ax.set_xticks(positions)
    ax.set_xticklabels([FAMILY_DISPLAY.get(f, f) for f in FAMILY_ORDER])
    style.format_axis(ax, ylabel=ylabel)
    ax.set_title(title, fontweight="bold")


def _fusion_legend(fig, figure_key: str = "patch") -> None:
    """Two entries, bottom centre — the only distinction colour is carrying."""
    from matplotlib.patches import Patch

    colors = style.fusion_colors(figure_key)
    handles = [
        Patch(facecolor=colors["image"], edgecolor="black", linewidth=0.8,
              label=BAR_LABELS["image"]),
        Patch(facecolor=colors["fusion"], edgecolor="black", linewidth=0.8,
              hatch="///", label=BAR_LABELS["fusion"]),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, -0.04),
    )


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def figure1_patch_f1(summary: pd.DataFrame, out_dir, name: str = "figure1_patch_f1") -> Path:
    """Patch-level F1 across the three rat VML tasks."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=style.FIGURE_SIZES["triptych"], sharey=True)

    for ax, task in zip(axes, TASK_ORDER):
        sub = _panel_frame(summary, task)
        _grouped_bars(
            ax, sub, "patch_f1",
            "F1-score" if ax is axes[0] else "",
            PANEL_TITLES[task],
            figure_key="patch",
        )

    _fusion_legend(fig, "patch")
    fig.tight_layout()

    caption = (
        "Figure 1. Patch-level F1-score across rat VML classification tasks. Left, Control "
        "versus VML; middle, Poor versus Very poor; right, three-class severity grading "
        "(Perfect/Poor/Very poor). Within each model family the left bar is the image-only "
        "model and the right, hatched bar is the same model fused with the two slide-level "
        "morphometric features (mean myofiber cross-sectional area and myofiber count). "
        "Binary panels report F1-score for the more injured class (VML; Very poor); the "
        "multiclass panel reports macro-averaged F1-score. Classes are ordered by severity "
        "throughout, so the positive class is always the more injured one. Models are logistic "
        "regression (LR), XGBoost, and ResNet-50, trained and evaluated on the class-balanced "
        "datasets."
    )
    return _save(fig, out_dir, name, caption)


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def figure2_roc(
    summary: pd.DataFrame, out_dir, name: str = "figure2_roc"
) -> Path:
    """ROC curves per task, read back from each run's predictions."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=style.FIGURE_SIZES["triptych"])

    for ax, task in zip(axes, TASK_ORDER):
        sub = _panel_frame(summary, task)
        binary = TASK_NUM_CLASSES[task] == 2
        for _, row in sub.iterrows():
            predictions = Path(row["source"])
            if not predictions.is_file():
                continue
            df = pd.read_csv(predictions)
            prob_cols = sorted(
                [c for c in df.columns if c.startswith("predicted_prob_class_")],
                key=lambda c: int(c.rsplit("_", 1)[-1]),
            )
            y_true = df["true_label"].to_numpy()
            y_proba = df[prob_cols].to_numpy()
            label_suffix = FUSION_SUFFIX if row["morphometrics"] else ""

            if binary:
                roc = binary_roc(y_true, y_proba[:, 1])
                if roc is None:
                    continue
                fpr, tpr, auc_value = roc
            else:
                _, macro_auc, _ = ovr_roc(y_true, y_proba, len(prob_cols), TASK_CLASS_NAMES[task])
                # Macro-average ROC over the one-vs-rest curves.
                grid = np.linspace(0, 1, 201)
                interpolated = []
                for i in range(len(prob_cols)):
                    y_bin = (y_true == i).astype(int)
                    if len(np.unique(y_bin)) < 2:
                        continue
                    roc = binary_roc(y_bin, y_proba[:, i])
                    if roc is None:
                        continue
                    interpolated.append(np.interp(grid, roc[0], roc[1]))
                if not interpolated:
                    continue
                fpr, tpr = grid, np.mean(interpolated, axis=0)
                auc_value = macro_auc if macro_auc is not None else float(np.trapz(tpr, fpr))

            line = style.roc_style(
                row["family"], bool(row["morphometrics"]), n_points=len(fpr)
            )
            family = FAMILY_DISPLAY.get(row["family"], row["family"])
            ax.plot(fpr, tpr, label=f"{family}{label_suffix} ({auc_value:.3f})", **line)

        ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.2, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(ROC_PANEL_TITLES[task], fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, frameon=True, edgecolor="0.7")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("ROC curves across classification tasks", fontweight="bold", y=1.02)
    fig.tight_layout()

    caption = (
        "Figure 2. Receiver operating characteristic curves across rat VML tasks (Control vs VML; "
        "Poor vs Very Poor; Perfect/Poor/Very Poor health). Marker shape identifies the model "
        "family and line style the input: solid with filled markers for image-only models, dashed "
        "with open markers for models fused with the two slide-level morphometric features, "
        "denoted by the suffix + morph. Binary panels show the ROC for the more injured class; "
        "the multiclass panel shows the macro-averaged one-vs-rest curve. Areas under the curve "
        "are listed in each panel. Models: LR, XGBoost, and ResNet-50."
    )
    return _save(fig, out_dir, name, caption)


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def figure3_wsi_f1(summary: pd.DataFrame, out_dir, name: str = "figure3_wsi_f1") -> Path:
    """Slide-level F1 after averaging patch probabilities within each WSI."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=style.FIGURE_SIZES["triptych"], sharey=True)

    for ax, task in zip(axes, TASK_ORDER):
        sub = _panel_frame(summary, task)
        n_wsi = int(sub["n_wsi"].max()) if not sub.empty else 0
        title = PANEL_TITLES[task]
        if n_wsi and SHOW_WSI_COUNT_IN_TITLE and title.endswith(")"):
            title = f"{title[:-1]}, {n_wsi} test WSIs)"
        _grouped_bars(
            ax, sub, "wsi_f1",
            "F1-score" if ax is axes[0] else "",
            title,
            figure_key="wsi",
        )

    _fusion_legend(fig, "wsi")
    fig.tight_layout()

    caption = (
        "Figure 3. WSI-level F1-score for LR, XGBoost, and ResNet-50 across three rat VML tasks, "
        "computed after averaging patch probabilities within each slide. Within each model "
        "family the left bar is the image-only model and the right, hatched bar is the same "
        "model fused with slide-level morphometrics. Binary panels report F1 for the more "
        "injured class (VML; Very Poor); multiclass reports macro-averaged F1. Test splits "
        "contain 8, 6 and 11 slides, so these values move in coarse steps."
    )
    return _save(fig, out_dir, name, caption)


# ── Figure 4 ─────────────────────────────────────────────────────────────────

ACLR_TASK_ORDER = ["uninjured_1mo", "uninjured_4mo", "1mo_4mo"]
ACLR_LABELS = {
    "uninjured_1mo": "Uninjured vs\n1-month post-ACLR",
    "uninjured_4mo": "Uninjured vs\n4-month post-ACLR",
    "1mo_4mo": "4-month vs\n1-month post-ACLR",
}


def figure4_aclr_transfer(
    metrics: pd.DataFrame,
    out_dir,
    name: str = "figure4_aclr_transfer",
    value_column: str = "accuracy",
) -> Path:
    """Rat-trained ResNet-50 applied to human VL histology following ACL injury, without retraining.

    ``metrics`` needs ``task_key``, ``cnn_only_<value_column>`` and
    ``cnn_morph_<value_column>`` columns.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZES["wide"])

    df = metrics.set_index("task_key")
    keys = [k for k in ACLR_TASK_ORDER if k in df.index]
    positions = np.arange(len(keys), dtype=float)
    width = 0.36

    for offset, prefix, morph in ((-width / 2, "cnn_only", False), (width / 2, "cnn_morph", True)):
        column = f"{prefix}_{value_column}"
        values = [float(df.loc[k, column]) for k in keys]
        bars = ax.bar(
            positions + offset, values, width=width,
            color=style.fusion_color("transfer", morph),
            hatch=style.fusion_hatch(morph), edgecolor="black", linewidth=0.8, zorder=3,
            label=BAR_LABELS["fusion"] if morph else BAR_LABELS["image"],
        )
        style.label_bars(ax, bars, values)

    # Chance level per comparison: 1/2 for the binary tasks, 1/3 for three-class.
    # Without it a reader cannot see that several image-only bars sit *below*
    # chance, which is the paper's point about limited direct transfer.
    if value_column == "accuracy":
        for i, key in enumerate(keys):
            chance = 0.5
            ax.plot(
                [positions[i] - 0.5, positions[i] + 0.5], [chance, chance],
                color="#666666", linestyle=":", linewidth=1.4,
                label="Chance" if i == 0 else None,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([ACLR_LABELS.get(k, k) for k in keys])
    ax.set_ylim(0, 1.05 if max(df.filter(like=value_column).max()) <= 1 else None)
    ax.set_ylabel("Patch-level accuracy" if value_column == "accuracy" else value_column)
    ax.set_title(
        "Exploratory transfer of rat-trained ResNet-50 to human VL histology",
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, edgecolor="0.8")
    fig.tight_layout()

    caption = (
        "Figure 4. Exploratory transfer of rat-trained models to human VL histology following ACL injury. "
        "Patch-level accuracy is shown for image-only and image-plus-morphometric ResNet-50 "
        "models across the binary and three-class human comparisons. No retraining or "
        "fine-tuning on human samples was performed. Absolute performance is well below the "
        "rat test sets, reflecting differences in species, muscle, staining and injury "
        "mechanism; morphometric fusion nevertheless improves accuracy in every comparison."
    )
    return _save(fig, out_dir, name, caption)


# ── Tables ───────────────────────────────────────────────────────────────────

def table1_datasets(bundles: Dict[str, Dict], out_dir, name: str = "table1_datasets") -> pd.DataFrame:
    """Dataset sizes per task, from live dataset scans.

    ``bundles`` maps task key to
    ``{"train_patches", "test_patches", "train_slides", "test_slides", ...}``
    as returned by :meth:`vmlevaluator.data.loaders.DataBundle.summary`.
    """
    rows = []
    for task in TASK_ORDER:
        info = bundles.get(task)
        if info is None:
            continue
        rows.append(
            {
                "Task": TASK_LABELS[task],
                "Classes": " / ".join(TASK_CLASS_NAMES[task]),
                "Train WSIs": info["train_slides"],
                "Test WSIs": info["test_slides"],
                "Total WSIs": info["train_slides"] + info["test_slides"],
                "Train patches": info["train_patches"],
                "Test patches": info["test_patches"],
                "Total patches": info["train_patches"] + info["test_patches"],
            }
        )
    table = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return table


def table2_ablation(
    summary: pd.DataFrame,
    out_dir,
    name: str = "table2_ablation",
    ablation_csv=None,
) -> pd.DataFrame:
    """Image-only vs morphometric-only vs fusion, patch and WSI F1.

    Isolates what each modality contributes: the morphometric-only row uses
    the same encoder and head as the fusion branch with no CNN backbone, so a
    gap between them is attributable to the image branch rather than capacity.

    The morphometric-only row is taken from ``summary`` when a ``mlp_morph``
    run is present. Otherwise it falls back to the recorded ablation table at
    ``results_reference/morphometric_only_ablation.csv``, and the ``Source``
    column says which was used — the ablation is a separate training sweep, so
    a reviewer regenerating only the six main models still gets a complete
    table with its provenance stated.
    """
    from .. import paths

    fallback = None
    csv_path = Path(ablation_csv or (paths.REFERENCE_DIR / "morphometric_only_ablation.csv"))
    if csv_path.is_file():
        fallback = pd.read_csv(csv_path)

    wanted = [
        ("cnn", "Image-only CNN"),
        ("mlp_morph", "Morphometric-only MLP"),
        ("cnn_morph", "CNN + morphometric fusion"),
    ]
    rows = []
    for task in TASK_ORDER:
        for model, label in wanted:
            match = summary[(summary["task"] == task) & (summary["model"] == model)]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "Task": TASK_LABELS[task],
                        "Model": label,
                        "Patch F1": round(float(row["patch_f1"]), 3),
                        "WSI F1": round(float(row["wsi_f1"]), 3),
                        "Source": "this run",
                    }
                )
                continue

            if model == "mlp_morph" and fallback is not None:
                hit = fallback[(fallback["task"] == task) & (fallback["model"] == "mlp")]
                if not hit.empty:
                    rows.append(
                        {
                            "Task": TASK_LABELS[task],
                            "Model": label,
                            "Patch F1": round(float(hit.iloc[0]["patch_f1_macro"]), 3),
                            "WSI F1": round(float(hit.iloc[0]["wsi_f1_macro"]), 3),
                            "Source": "recorded ablation",
                        }
                    )
                    continue

            rows.append(
                {"Task": TASK_LABELS[task], "Model": label,
                 "Patch F1": np.nan, "WSI F1": np.nan, "Source": "not available"}
            )

    table = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return table


def build_all(
    summary: pd.DataFrame,
    out_dir,
    *,
    aclr_metrics: Optional[pd.DataFrame] = None,
    dataset_bundles: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Path]:
    """Regenerate every figure and table that the available data supports."""
    out_dir = Path(out_dir)
    produced: Dict[str, Path] = {}

    if not summary.empty:
        produced["figure1"] = figure1_patch_f1(summary, out_dir)
        produced["figure2"] = figure2_roc(summary, out_dir)
        produced["figure3"] = figure3_wsi_f1(summary, out_dir)
        produced["table2"] = out_dir / "table2_ablation.csv"
        table2_ablation(summary, out_dir)
    if aclr_metrics is not None and not aclr_metrics.empty:
        produced["figure4"] = figure4_aclr_transfer(aclr_metrics, out_dir)
    if dataset_bundles:
        table1_datasets(dataset_bundles, out_dir)
        produced["table1"] = out_dir / "table1_datasets.csv"

    return produced

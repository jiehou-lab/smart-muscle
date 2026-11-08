#!/usr/bin/env python
"""Manuscript Figures 1-4 for the class-balanced dataset, drawn with the
package's own figure functions (the ones notebooks/figures/*.ipynb call), so the theme is
the manuscript's. Nothing in the package is edited; its model list, panel titles, bar labels
and ROC legend are overridden for this process only.

    python manuscript_figures.py                       # -> runs/figures/figure{1,2,3,4}*.png/.pdf
    python manuscript_figures.py --numbers FILE --results-base DIR --out DIR

Figures 1 and 3: F1 for the more injured class (VML; Very Poor) on the binary tasks, macro-F1
on three-class (patch / WSI). Figure 2: ROC (binary panels: the more injured class; three-class:
macro one-vs-rest), one shared legend, each panel's AUCs in its lower-right corner.
Models: LR (CV-tuned), XGBoost, ResNet-50, each image only and fused with morphometrics.
Figure 4: rat-trained ResNet-50 on human VL tissue following ACL injury, three binary
comparisons, F1 for the uninjured group (4-month in 4-month vs 1-month) at patch and WSI level.

Ported from the manuscript run (the manuscript run's make_manuscript_figures.py); only the paths differ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REFERENCE, WORK, roots   # noqa: E402  (also puts the package on sys.path)
from vmlevaluator.figures import paper_figures as pf, style   # noqa: E402

FAM = {"lr": "LR", "xgb": "XGB", "cnn": "ResNet50"}
BAR_VALUE_SIZE = 11       # package default 9
BAR_VALUE_NUDGE = 3       # points; the two values of a pair move apart so equal-height neighbours never touch
ROC_LEGEND_SIZE = 12.5
ROC_AUC_SIZE = 11


def _label_bars(ax, bars, values, fmt="{:.3f}"):
    """The package's bar labels (bold, just above the bar), larger, to three decimals -- formatted
    exactly as the values quoted in the manuscript text. Bars sit either side of integer x positions;
    a left-hand bar's value is nudged left and a right-hand bar's value right."""
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        dx = -BAR_VALUE_NUDGE if x < round(x) else BAR_VALUE_NUDGE
        ax.annotate(fmt.format(value), (x, bar.get_height() + 0.012), xytext=(dx, 0), textcoords="offset points",
                    ha="center", va="bottom", fontsize=BAR_VALUE_SIZE, fontweight="bold", color="#222222")


def _save_roc_shared_legend(fig, out_dir, name, caption, _save=pf._save):
    """Figure 2: the package gives each panel its own 8 pt legend, "model (AUC)", in the lower-right
    corner. Replace them with one shared legend (line styles only) below the panels, and list each
    panel's AUCs in its lower-right corner -- below the chance diagonal, where no ROC curve runs."""
    if name == "figure2_roc":
        shared = None
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            handles, labels = ax.get_legend_handles_labels()
            leg.remove()
            names = [re.sub(r" \([0-9.]+\)$", "", l) for l in labels]
            shared = shared or (handles, names)
            rows = [(n, re.search(r"\(([0-9.]+)\)$", l).group(1), h.get_color())
                    for n, l, h in zip(names, labels, handles) if n != "Chance"]
            y = 0.03
            for n, auc, col in reversed(rows):
                ax.text(0.97, y, f"{n}  {auc}", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=ROC_AUC_SIZE, color=col, fontweight="bold")
                y += 0.068
            ax.text(0.97, y, "AUC", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=ROC_AUC_SIZE, color="#222222", fontweight="bold")
        fig.legend(*shared, loc="lower center", ncol=len(shared[0]), fontsize=ROC_LEGEND_SIZE, frameon=True,
                   edgecolor="0.7", handlelength=2.6, columnspacing=1.4, bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
    return _save(fig, out_dir, name, caption)


def summary_table(N: dict, base=None) -> pd.DataFrame:
    """The package's summary frame: one row per (task, model), F1 as the manuscript reports it,
    and the predictions.csv the ROC curves are drawn from."""
    R = roots(base)
    rows = []
    for t in pf.TASK_ORDER:
        for fam in ("lr", "xgb", "cnn"):
            for m in (fam, fam + "_morph"):
                v = N["rat"][f"{m}__{t}"]
                rows.append(dict(task=t, model=m, family=FAM[fam], morphometrics=m.endswith("_morph"),
                                 # binary tasks: F1 for the more injured class (VML; Very Poor); three-class: macro-F1
                                 patch_f1=v["patch_macro" if t == "multiclass" else "patch_f1_pos"],
                                 wsi_f1=v["wsi_macro" if t == "multiclass" else "wsi_f1_pos"], n_wsi=v["n_wsi"],
                                 source=str(R[fam] / f"{m}__{t}" / "evaluation" / "predictions.csv")))
    return pd.DataFrame(rows)


def make_figures(N: dict | None = None, out=None, base=None) -> list:
    """Draw Figures 1-4 into ``out`` from the numbers ``N`` (default: the manuscript's frozen numbers)
    and the prediction tables under ``base`` (ROC curves). Returns the saved paths."""
    N = N if N is not None else json.loads((REFERENCE / "manuscript_numbers.json").read_text())
    out = Path(out) if out is not None else WORK / "figures"
    summary = summary_table(N, base)

    pf.FAMILY_ORDER = ["LR", "XGB", "ResNet50"]
    pf.FIGURE_MODELS = ["lr", "lr_morph", "xgb", "xgb_morph", "cnn", "cnn_morph"]
    # Plain one-line panel titles (the caption says which F1 is shown), no WSI count, "+ morph" in the ROC legend.
    pf.SHOW_WSI_COUNT_IN_TITLE = False
    pf.PANEL_TITLES = {"control_vml": "Control vs VML", "poor_very_poor": "Poor vs Very poor", "multiclass": "Multiclass"}
    pf.FUSION_SUFFIX = " + morph"
    # Bar-figure legends: "Fusion" alone (package: "Fusion (image + morphometrics)"), as in Figure 4.
    pf.BAR_LABELS = dict(pf.BAR_LABELS, fusion="Fusion")
    style.label_bars = _label_bars
    if getattr(pf._save, "__name__", "") != "_save_roc_shared_legend":
        pf._save = _save_roc_shared_legend

    # Figure 2 (ROC) only: larger titles, axis labels and ticks (package: 13 / 12 / 10).
    bar_typo = dict(style.TYPOGRAPHY)
    roc_typo = dict(bar_typo, title=15, label=14, tick=12.5)

    f1 = pf.figure1_patch_f1(summary, out, name="figure1_patch_f1")
    style.TYPOGRAPHY.update(roc_typo)
    try:
        f2 = pf.figure2_roc(summary, out, name="figure2_roc")
    finally:
        style.TYPOGRAPHY.update(bar_typo)
    f3 = pf.figure3_wsi_f1(summary, out, name="figure3_wsi_f1")

    # Figure 4: same bar style as the package's figure4_aclr_transfer (transfer colours, hatching,
    # bar labels), two panels -- patch and WSI -- and the three binary comparisons.
    plt = pf._plt()
    keys = ["uninjured_1mo", "uninjured_4mo", "1mo_4mo"]
    labels = {"uninjured_1mo": "Uninjured vs\n1-month post-ACLR", "uninjured_4mo": "Uninjured vs\n4-month post-ACLR",
              "1mo_4mo": "4-month vs\n1-month post-ACLR"}
    fig, axes = plt.subplots(1, 2, figsize=(style.FIGURE_SIZES["wide"][0] * 1.5, style.FIGURE_SIZES["wide"][1]), sharey=True)
    width, pos = 0.36, np.arange(len(keys), dtype=float)
    for ax, lvl, title in ((axes[0], "patch", "Patch level"), (axes[1], "wsi", "WSI level")):
        for offset, m, morph, lab in ((-width / 2, "cnn", False, "ResNet-50"), (width / 2, "cnn_morph", True, "Fusion")):
            vals = [N["human"][f"{m}__{k}"][f"{lvl}_f1"] for k in keys]
            # Colours of the manuscript's original Figure 4: blue image-only, orange (hatched) fused.
            bars = ax.bar(pos + offset, vals, width=width, color="#E07A3D" if morph else "#2E5A88",
                          hatch=style.fusion_hatch(morph), edgecolor="black", linewidth=0.8, zorder=3, label=lab)
            style.label_bars(ax, bars, vals)
        ax.set_xticks(pos)
        ax.set_xticklabels([labels[k] for k in keys])
        style.format_axis(ax, ylabel="F1-score" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=True, edgecolor="0.7", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    f4 = pf._save(fig, out, "figure4_aclr_transfer", "")
    return [f1, f2, f3, f4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", default=None, help="numbers JSON (default: reference/manuscript_numbers.json)")
    ap.add_argument("--results-base", default=None, help="folder holding the neural_networks* roots")
    ap.add_argument("--out", default=str(WORK / "figures"))
    args = ap.parse_args()
    N = json.loads(Path(args.numbers).read_text()) if args.numbers else None
    for f in make_figures(N, args.out, args.results_base):
        print(f)


if __name__ == "__main__":
    main()

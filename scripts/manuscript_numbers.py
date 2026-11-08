#!/usr/bin/env python
"""Every number quoted in the manuscript, from the prediction tables.

    python manuscript_numbers.py                       # -> runs/manuscript_numbers.json, compared with reference/
    python manuscript_numbers.py --results-base DIR    # the same, from a re-run under DIR

Rat (class-balanced test set): F1 for the more injured class (VML; Very Poor), macro-F1
and accuracy, patch and WSI; ROC-AUC computed exactly as Figure 2 draws it (binary: the
more injured class; three-class: macro one-vs-rest). Models: LR (CV-tuned), XGBoost,
ResNet-50, each image only and + morph; MLP morph-only for the ablation table. Human
(binary comparisons, ResNet-50 only): F1 for Uninjured (4-month in 4-month vs 1-month),
macro-F1, accuracy and patch-level ROC-AUC.

WSI level = patch probabilities averaged per slide, then argmax (VMLevaluator's rule).
Ported from the manuscript run's collect.py (metr) and manuscript_numbers.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HUMAN, REFERENCE, TASKS, WORK, roots   # noqa: E402

MODELS = ["lr", "lr_morph", "xgb", "xgb_morph", "cnn", "cnn_morph", "mlp_morph"]


def metr(path: Path):
    d = pd.read_csv(path)
    pc = sorted(c for c in d.columns if c.startswith("predicted_prob_class_"))
    k = len(pc)
    d["w"] = d.filename.str.split("__r").str[0]
    g = d.groupby("w")
    out = {}
    for lvl, (yt, yp) in {"patch": (d.true_label.values, d.predicted_label.values),
                          "wsi": (g.true_label.first().values, g[pc].mean().values.argmax(1))}.items():
        per = f1_score(yt, yp, average=None, labels=list(range(k)), zero_division=0)
        out[lvl] = dict(f1=[round(float(x), 4) for x in per], macro=round(float(per.mean()), 4),
                        acc=round(float(accuracy_score(yt, yp)), 4), n=int(len(yt)),
                        true=np.bincount(yt, minlength=k).tolist(), pred=np.bincount(yp, minlength=k).tolist())
    return out


def pred(m, t, human=None, base=None):
    r = roots(base)[m.split("_")[0]]
    return r / ("human/%s/%s" % (human, m) if human else f"{m}__{t}/evaluation") / "predictions.csv"


def auc(p, multiclass):
    d = pd.read_csv(p)
    pc = sorted(c for c in d.columns if c.startswith("predicted_prob_class_"))
    y = d.true_label.values
    if multiclass:
        return float(np.mean([roc_auc_score(y == k, d[c]) for k, c in enumerate(pc)]))
    return float(roc_auc_score(y == 1, d[pc[1]]))


def compute(base=None, models=MODELS, tasks=TASKS, human=HUMAN):
    """{"rat": {model__task: {...}}, "human": {model__comparison: {...}}} for every run found under ``base``."""
    out = {"rat": {}, "human": {}}
    for t in tasks:
        for m in models:
            p = pred(m, t, base=base)
            if not p.is_file():
                continue
            r = metr(p)
            out["rat"][f"{m}__{t}"] = dict(patch_macro=r["patch"]["macro"], wsi_macro=r["wsi"]["macro"],
                                           # F1 for the more injured class (VML; Very Poor), the binary-task convention
                                           patch_f1_pos=r["patch"]["f1"][-1], wsi_f1_pos=r["wsi"]["f1"][-1],
                                           patch_acc=r["patch"]["acc"], wsi_acc=r["wsi"]["acc"],
                                           auc=auc(p, t == "multiclass"), n_wsi=r["wsi"]["n"], n_patch=r["patch"]["n"])
    for c in human:
        for m in ("cnn", "cnn_morph"):
            p = pred(m, None, human=c, base=base)
            if not p.is_file():
                continue
            r = metr(p)
            d = pd.read_csv(p)
            out["human"][f"{m}__{c}"] = dict(patch_f1=r["patch"]["f1"][0], wsi_f1=r["wsi"]["f1"][0],
                                             patch_macro=r["patch"]["macro"], wsi_macro=r["wsi"]["macro"],
                                             patch_acc=r["patch"]["acc"], wsi_acc=r["wsi"]["acc"],
                                             auc=float(roc_auc_score(d.true_label == 0, d.predicted_prob_class_0)),
                                             wsi_called0=r["wsi"]["pred"][0], wsi_true0=r["wsi"]["true"][0], n_wsi=r["wsi"]["n"])
    return out


def compare(new: dict, ref: dict | None = None, decimals: int = 3) -> pd.DataFrame:
    """One row per number in the manuscript: reference value, reproduced value, and whether they
    agree at the precision the manuscript prints (``decimals``)."""
    ref = ref if ref is not None else json.loads((REFERENCE / "manuscript_numbers.json").read_text())
    rows = []
    for part in ("rat", "human"):
        for key, vals in ref[part].items():
            got = new.get(part, {}).get(key)
            for metric, v in vals.items():
                g = None if got is None else got.get(metric)
                ok = g is not None and round(float(g), decimals) == round(float(v), decimals)
                rows.append(dict(part=part, run=key, metric=metric, manuscript=v, reproduced=g,
                                 diff=None if g is None else float(g) - float(v),
                                 status="match" if ok else ("missing" if g is None else "DIFFERS")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", default=None, help="folder holding the neural_networks* roots")
    ap.add_argument("--out", default=str(WORK / "manuscript_numbers.json"))
    args = ap.parse_args()
    out = compute(args.results_base)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["rat"].items():
        print(f"{k:28s} F1 injured {v['patch_f1_pos']:.3f}/{v['wsi_f1_pos']:.3f} | patch macro {v['patch_macro']:.3f} WSI macro {v['wsi_macro']:.3f} | acc {v['patch_acc']:.3f}/{v['wsi_acc']:.3f} | AUC {v['auc']:.3f}")
    for k, v in out["human"].items():
        print(f"{k:24s} F1 {v['patch_f1']:.3f}/{v['wsi_f1']:.3f} macro {v['patch_macro']:.3f}/{v['wsi_macro']:.3f} acc {v['patch_acc']:.3f}/{v['wsi_acc']:.3f} AUC {v['auc']:.3f} called0 {v['wsi_called0']}/{v['n_wsi']} (true {v['wsi_true0']})")
    cmp = compare(out)
    print(f"\n{(cmp.status == 'match').sum()} of {len(cmp)} manuscript numbers reproduced to 3 decimals")
    bad = cmp[cmp.status != "match"]
    if len(bad):
        print(bad.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

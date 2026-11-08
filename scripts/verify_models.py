#!/usr/bin/env python
"""Load every saved model behind the manuscript, re-score the rat test split with it,
and check that the predictions are the ones the manuscript was written from.

    python verify_models.py                    # all 21 models (~15 min with a GPU, longer on CPU)
    python verify_models.py --models lr,xgb    # a subset
    python verify_models.py --human            # also re-score the human comparisons (ResNet-50)

Each model is loaded from ``models/`` with its own weights and normalisation statistics,
run through the package's evaluation code on ``data/Dataset_balanced/<task>/test``, and its
per-patch predicted labels and class probabilities are compared with the shipped
``predictions.csv``. Output goes to ``runs/verify_models/``; ``models/`` is not touched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, HUMAN, HUMAN_ROOT, HUMAN_WB, MODELS, SYNTHETIC, TASKS, WORK, roots   # noqa: E402

TOL = 1e-4      # probabilities; labels must match exactly
HUMAN_NAMES = {"uninjured_1mo": ("control_vml", ["Uninjured", "1-month post-ACLR"]),
               "uninjured_4mo": ("control_vml", ["Uninjured", "4-month post-ACLR"]),
               "1mo_4mo": ("poor_very_poor", ["4-month post-ACLR", "1-month post-ACLR"])}


def config(task, model, res=None):
    """The experiment config the model was trained with. XGBoost runs through the package's
    classical (``rf``) pipeline with the estimator swapped, so its config is the rf one."""
    from vmlevaluator.config import load_experiment
    res = roots()[model.split("_")[0]] if res is None else res
    if model.startswith("xgb"):
        import run_xgb_shim  # noqa: F401  (the saved XGBoost bundles reference this class)
        return load_experiment(task, model.replace("xgb", "rf"), data_root=DATA, results_root=res,
                               output_name=f"{model}__{task}")
    return load_experiment(task, model, data_root=DATA, results_root=res)


def same(a: Path, b: Path):
    x = pd.read_csv(a).sort_values("filename").reset_index(drop=True)
    y = pd.read_csv(b).sort_values("filename").reset_index(drop=True)
    pc = [c for c in y.columns if c.startswith("predicted_prob_class_")]
    if len(x) != len(y) or not (x.filename == y.filename).all():
        return False, float("nan"), len(y)
    diff = float(abs(x[pc].values - y[pc].values).max())
    return bool((x.predicted_label == y.predicted_label).all()) and diff < TOL, diff, len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma list of models or families (default: all)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--human", action="store_true", help="also re-score the human comparisons (cnn, cnn_morph)")
    args = ap.parse_args()
    from vmlevaluator.evaluate.runner import evaluate_experiment
    want = args.models.split(",") if args.models else MODELS
    models = [m for m in MODELS if m in want or m.split("_")[0] in want]
    out_root = WORK / "verify_models"
    rows = []
    jobs = [(t, m, None) for t in args.tasks.split(",") for m in models]
    if args.human:
        jobs += [(t, m, c) for c, (t, _) in HUMAN_NAMES.items() for m in ("cnn", "cnn_morph") if m in models]
    for task, model, comp in jobs:
        t0 = time.time()
        cfg = config(task, model)
        if comp is None:
            ref = roots()[model.split("_")[0]] / f"{model}__{task}" / "evaluation" / "predictions.csv"
            out = out_root / f"{model}__{task}"
        else:
            cfg.data.test.img_path = str(HUMAN_ROOT / comp)
            cfg.data.class_names = HUMAN_NAMES[comp][1]
            cfg.data.folder_to_class = [0, 1]
            if cfg.data.get("morphometrics_path") is not None:
                cfg.data.morphometrics_path = str(HUMAN_WB)
            ref = roots()["cnn"] / "human" / comp / model / "predictions.csv"
            out = out_root / "human" / comp / model
        evaluate_experiment(cfg, output_dir=str(out), make_plots=False, verbose=False)
        ok, diff, n = same(out / "predictions.csv", ref)
        name = f"{model}__{task}" if comp is None else f"{model} on human {comp}"
        rows.append(dict(model=model, data=task if comp is None else f"human/{comp}", patches=n,
                         labels_and_probs_match=ok, max_prob_diff=diff, seconds=round(time.time() - t0)))
        print(f"{'match   ' if ok else 'MISMATCH'} {name:36s} {n:5d} patches, max |p diff| {diff:.1e} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "verification.csv", index=False)
    print(f"\n{int(df.labels_and_probs_match.sum())} of {len(df)} saved models reproduce their {'saved' if SYNTHETIC else 'manuscript'} predictions"
          f" -> {out_root / 'verification.csv'}")
    sys.exit(0 if df.labels_and_probs_match.all() else 1)


if __name__ == "__main__":
    main()

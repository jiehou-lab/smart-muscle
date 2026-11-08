#!/usr/bin/env python
"""Score every trained model on the human VL comparison folders (following ACL injury), zero-shot.

    python human_eval.py --data-root F:\\...\\Dataset_balanced --results-root F:\\...\\models
    python human_eval.py ... --selftest     # prove the path reproduces the rat test predictions first

Each trained experiment ``<model>__<task>`` in the results root is evaluated
with its own weights and its own rat normalisation statistics, but with the
test folder pointed at a human comparison folder and the morphometrics
workbook pointed at the human one. Nothing is retrained. Output goes to
``<results-root>/human/<comparison>/<model>/`` -- the rat evaluation folders
are not touched.

Comparison -> rat model used:
    uninjured_1mo, uninjured_4mo  ->  control_vml
    1mo_4mo                       ->  poor_very_poor
All human folders are severity-ordered (class index = folder integer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HUMAN_ROOT, HUMAN_WB as HUMAN_WORKBOOK   # noqa: E402  (also puts the package on sys.path)
COMPARISONS = {   # the three human comparisons in the manuscript
    "uninjured_1mo": ("control_vml", ["Uninjured", "1-month post-ACLR"]),
    "uninjured_4mo": ("control_vml", ["Uninjured", "4-month post-ACLR"]),
    "1mo_4mo": ("poor_very_poor", ["4-month post-ACLR", "1-month post-ACLR"]),
}


def trained(results_root: Path):
    for d in sorted(results_root.glob("*__*")):
        if (d / "evaluation" / "metrics.json").is_file():
            model, task = d.name.split("__", 1)
            yield model, task


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", default=None, help="comma list; default every trained model")
    ap.add_argument("--swap", action="store_true",
                    help="models trained on a class-reversed dataset (Dataset_balanced_swapped): read the human "
                         "folders in reverse too, so Uninjured (and 4-month in 4-month vs 1-month) is the last class")
    args = ap.parse_args()

    import pandas as pd
    from vmlevaluator.config import load_experiment
    from vmlevaluator.evaluate.runner import evaluate_experiment

    rroot = Path(args.results_root)
    want = set(args.models.split(",")) if args.models else None
    runs = [(m, t) for m, t in trained(rroot) if want is None or m in want]

    if args.selftest:
        # Same code path, rat test folder: predictions must match the sweep's own.
        bad = 0
        for model, task in runs:
            cfg = load_experiment(task, model, data_root=Path(args.data_root), results_root=rroot)
            out = rroot / "_selftest" / f"{model}__{task}"
            evaluate_experiment(cfg, output_dir=str(out), make_plots=False, verbose=False)
            a = pd.read_csv(out / "predictions.csv").sort_values("filename").reset_index(drop=True)
            b = pd.read_csv(rroot / f"{model}__{task}" / "evaluation" / "predictions.csv").sort_values("filename").reset_index(drop=True)
            pc = [c for c in b.columns if c.startswith("predicted_prob_class_")]
            ok = len(a) == len(b) and (a.filename == b.filename).all() and \
                (a.predicted_label == b.predicted_label).all() and (abs(a[pc].values - b[pc].values).max() < 1e-4)
            bad += not ok
            print(f"selftest {model}__{task}: {'match' if ok else 'MISMATCH'}", flush=True)
        sys.exit(1 if bad else 0)

    done = []
    for comp, (task, names) in COMPARISONS.items():
        folder = HUMAN_ROOT / comp
        if not folder.is_dir():
            print(f"{comp}: no folder at {folder}; skipped")
            continue
        for model, t in runs:
            if t != task:
                continue
            out = rroot / "human" / comp / model
            if (out / "metrics.json").is_file():
                print(f"[skip] {comp} / {model}", flush=True)
                continue
            cfg = load_experiment(task, model, data_root=Path(args.data_root), results_root=rroot)
            cfg.data.test.img_path = str(folder)
            k = len(names)
            if args.swap:   # human folder i -> class k-1-i, matching the reversed rat classes
                cfg.data.class_names = list(reversed(names))
                cfg.data.folder_to_class = [k - 1 - i for i in range(k)]
            else:
                cfg.data.class_names = names
                cfg.data.folder_to_class = list(range(k))
            if cfg.data.get("morphometrics_path") is not None:
                cfg.data.morphometrics_path = str(HUMAN_WORKBOOK)
            try:
                r = evaluate_experiment(cfg, output_dir=str(out), make_plots=False, verbose=False)
                print(f"[done] {comp:18s} {model:14s} patch acc {r.patch.accuracy:.3f}  "
                      f"WSI acc {r.wsi.accuracy:.3f}", flush=True)
                done.append((comp, model))
            except Exception as exc:
                print(f"[FAIL] {comp} / {model}: {type(exc).__name__}: {exc}", flush=True)
    (rroot / "human" / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

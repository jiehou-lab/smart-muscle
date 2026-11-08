#!/usr/bin/env python
"""XGBoost on the class-balanced dataset.

    python run_xgb.py            # train + evaluate xgb / xgb_morph on all three tasks, then score the human comparisons

Runs through the package's classical (``rf``) pipeline -- same features (flattened ImageNet-normalised 3x224x224
pixels, plus z-scored mean CSA and fibre count for the "+ morph" variant), same
StandardScaler, same Dataset_balanced train/test split -- with only the estimator
swapped for xgboost.XGBClassifier. Hyper-parameters are fixed in advance (not
tuned on test): 100 trees, depth 4, 32 histogram bins, learning rate 0.3, 80% rows and
10% features per tree, hist on the GPU when available (sized to finish in time). The classes are balanced,
so no class weights.

Results: <VML_RESULTS>/xgboost/{xgb,xgb_morph}__<task>/
and .../human/<comparison>/<model>/. Nothing in VMLevaluator is edited; its
estimator factory is replaced for this process only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import DATA, HUMAN_ROOT, HUMAN_WB, roots   # noqa: E402  (also puts the package on sys.path)
RES = roots()["xgb"]
COMPARISONS = {   # the three human comparisons in the manuscript
    "uninjured_1mo": ("control_vml", ["Uninjured", "1-month post-ACLR"]),
    "uninjured_4mo": ("control_vml", ["Uninjured", "4-month post-ACLR"]),
    "1mo_4mo": ("poor_very_poor", ["4-month post-ACLR", "1-month post-ACLR"]),
}
PAIRS = [("xgb", "rf"), ("xgb_morph", "rf_morph")]   # our name -> the RF config it mirrors

import vmlevaluator.train.sklearn_trainer as st        # noqa: E402
from vmlevaluator.config import load_experiment        # noqa: E402
from vmlevaluator.evaluate.runner import evaluate_experiment  # noqa: E402

_orig_build = st.build_estimator


def _device():
    try:
        import xgboost as xgb
        import numpy as np
        xgb.XGBClassifier(n_estimators=1, device="cuda", tree_method="hist").fit(np.zeros((4, 2)), [0, 1, 0, 1])
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = "cpu"   # the 150k-feature matrix does not fit next to the display on the 8 GB GPU (cudaErrorMemoryAllocation)


from run_xgb_shim import XGBForestShim  # noqa: E402


def build_xgb(cfg, num_classes, n_features):
    if str(cfg.family).upper() != "RF":
        return _orig_build(cfg, num_classes, n_features)
    return XGBForestShim(
        n_estimators=100, max_depth=4, max_bin=32, learning_rate=0.3, subsample=0.8, colsample_bytree=0.1,
        tree_method="hist", device=DEVICE, random_state=int(cfg.environment.seed), n_jobs=8,
        objective="binary:logistic" if num_classes == 2 else "multi:softprob",
        eval_metric="logloss" if num_classes == 2 else "mlogloss",
    )


st.build_estimator = build_xgb   # train() looks the factory up at call time


def cfg_for(task, ours, rf_model):
    cfg = load_experiment(task, rf_model, data_root=DATA, results_root=RES, output_name=f"{ours}__{task}",
                          overrides=["streaming.max_in_memory_gb=8"])   # in-memory fit: the forest-merging stream path is RF-only
    # meta.model stays rf / rf_morph: the package keys its feature pipeline on it. The folder name says xgb.
    cfg.meta.estimator = "XGBClassifier"
    return cfg


def main(res=None):
    """Train + evaluate into ``res`` (default: the xgb results root), then score the human folders."""
    global RES
    RES = Path(res) if res is not None else RES
    print(f"xgboost device: {DEVICE}", flush=True)
    # One task at a time, its human comparisons straight after, so results arrive in order of use.
    for task in ("control_vml", "poor_very_poor", "multiclass"):
        for ours, rf_model in PAIRS:
            out = RES / f"{ours}__{task}"
            if (out / "evaluation" / "metrics.json").is_file():
                print(f"[skip] {ours}__{task}", flush=True)
                continue
            t0 = time.time()
            cfg = cfg_for(task, ours, rf_model)
            st.train_sklearn_model(cfg, verbose=True)
            r = evaluate_experiment(cfg, make_plots=False, verbose=False)
            print(f"[done] {ours}__{task}: patch acc {r.patch.accuracy:.3f}, WSI acc {r.wsi.accuracy:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        for comp, (t, names) in COMPARISONS.items():
            # Only the comparisons present under HUMAN_ROOT are scored (this package ships the
            # three used in the manuscript). XGBoost human scores are not reported in it.
            if t != task or not (HUMAN_ROOT / comp).is_dir():
                continue
            for ours, rf_model in PAIRS:
                out = RES / "human" / comp / ours
                if (out / "metrics.json").is_file():
                    continue
                cfg = cfg_for(task, ours, rf_model)
                cfg.data.test.img_path = str(HUMAN_ROOT / comp)
                cfg.data.class_names = names
                cfg.data.folder_to_class = list(range(len(names)))
                if cfg.data.get("morphometrics_path") is not None:
                    cfg.data.morphometrics_path = str(HUMAN_WB)
                r = evaluate_experiment(cfg, output_dir=str(out), make_plots=False, verbose=False)
                print(f"[human] {comp:18s} {ours:10s} patch acc {r.patch.accuracy:.3f}, WSI acc {r.wsi.accuracy:.3f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

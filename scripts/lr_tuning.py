#!/usr/bin/env python
"""Logistic-regression search on the class-balanced dataset (raw pixels, + 2 morph columns for
lr_morph, as in the paper), by animal-grouped 3-fold CV inside the training split.
The test split is read once, after the choice.

    python lr_tuning.py extract    # cache train features per task (pixels + 2 morph columns) -> runs/feature_cache
    python lr_tuning.py cv         # grid x 3 folds -> runs/lr_tuning_cv.csv
    python lr_tuning.py final      # refit the choice on all train -> logistic_regression (rat test)

Searched: inverse regularisation strength C (the published model uses 1.0 and reaches
100% training accuracy), and for the three-class task a weight on Perfect.

Selection, per task and model: highest CV patch F1 for the reported class (Control /
Poor / Perfect) among settings whose CV patch macro F1 is at least that of the
published setting (C = 1, no extra weight) -- so a setting cannot win by calling every
patch that class; CV WSI F1 for that class, then macro F1, break ties.
Chosen for the manuscript: C = 1e-6 for every task and model, Perfect weight 4
(reference/lr_tuned_params.json, from reference/lr_tuning_cv.csv).

Ported from the manuscript run's tune_rf.py (extract, folds, scores) and tune_lr.py
(cv, choose, final); only the paths differ.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, REFERENCE, WORK, roots   # noqa: E402  (also puts the package on sys.path)

RES = roots()["lr"]
CACHE = Path(os.environ.get("VML_CACHE", str(WORK / "feature_cache")))
OUT = WORK / "lr_tuning_cv.csv"
TASKS = ["control_vml", "poor_very_poor", "multiclass"]
VARIANTS = ["lr", "lr_morph"]
N_PIX = 3 * 224 * 224
SEED = 95
CS = [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
W = {"control_vml": [1], "poor_very_poor": [1], "multiclass": [1, 2, 4]}


def extract():
    from vmlevaluator.config import load_experiment
    from vmlevaluator.data.loaders import build_datasets
    from vmlevaluator.features import extract_features
    from vsplit import animal, extract_wsi_id
    CACHE.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        if (CACHE / f"{t}_X.npy").is_file():
            print(f"[skip] {t}", flush=True)
            continue
        cfg = load_experiment(t, "rf_morph", data_root=DATA, results_root=WORK, output_name="_extract")
        b = build_datasets(cfg, train_augmentation=False, with_validation=False, verbose=True)
        X, y, _ = extract_features(b.train, batch_size=64, num_workers=0, verbose=True)
        slides = [extract_wsi_id(p) for p in b.train.imgs]
        np.save(CACHE / f"{t}_X.npy", X.astype(np.float32, copy=False))
        pd.DataFrame(dict(y=y, slide=slides, animal=[animal(s) for s in slides])).to_csv(CACHE / f"{t}_meta.csv", index=False)
        print(f"[cached] {t}: {X.shape}", flush=True)
        del X


def folds(meta):
    from sklearn.model_selection import StratifiedGroupKFold
    k = len(set(meta.y))
    for seed in range(SEED, SEED + 50):   # first seeded split whose every fold holds every class
        f = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(meta, meta.y, meta.animal))
        if all(len(set(meta.y.iloc[va])) == k and len(set(meta.y.iloc[tr])) == k for tr, va in f):
            return f, seed
    raise RuntimeError("no 3-fold animal split with every class in every fold")


def scores(meta_va, proba):
    from sklearn.metrics import f1_score
    k = proba.shape[1]
    yp = proba.argmax(1)
    pf = f1_score(meta_va.y, yp, average=None, labels=list(range(k)), zero_division=0)
    d = pd.DataFrame(proba).assign(slide=meta_va.slide.values, y=meta_va.y.values).groupby("slide")
    wy, wp = d.y.first().values, d[list(range(k))].mean().values.argmax(1)
    wf = f1_score(wy, wp, average=None, labels=list(range(k)), zero_division=0)
    return dict(patch_macro=pf.mean(), patch_f1_0=pf[0], wsi_macro=wf.mean(), wsi_f1_0=wf[0])


def cw(w, y, k):
    y = np.asarray(y)
    n, c = len(y), np.bincount(y, minlength=k)
    return {j: float(n / (k * c[j])) * (float(w) if j == 0 else 1.0) for j in range(k)}


def cv():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(OUT).to_dict("records") if OUT.is_file() else []
    seen = {(r["task"], r["variant"], str(r["C"]), str(r["w"])) for r in rows}
    for t in TASKS:
        X = np.load(CACHE / f"{t}_X.npy", mmap_mode="r")
        meta = pd.read_csv(CACHE / f"{t}_meta.csv")
        k = int(meta.y.max()) + 1
        f, _ = folds(meta)
        for v in VARIANTS:
            Xv = np.ascontiguousarray(X[:, :N_PIX] if v == "lr" else X)
            for C, w in itertools.product(CS, W[t]):
                if (t, v, str(C), str(w)) in seen:
                    continue
                # Time: three-class fits at C >= 0.1 take ~30 min each and were no better for lr; for lr_morph keep
                # only the published baseline (C = 1, w = 1) there.
                if t == "multiclass" and v == "lr_morph" and (C == 0.1 or (C == 1.0 and w != 1)):
                    continue
                t0, iters = time.time(), []
                oof = np.zeros((len(meta), k), dtype=np.float32)
                for tr, va in f:
                    sc = StandardScaler().fit(Xv[tr])
                    m = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-4,
                                           class_weight=cw(w, meta.y.values[tr], k), random_state=SEED)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        m.fit(sc.transform(Xv[tr]).astype(np.float32), meta.y.values[tr])
                    oof[va] = m.predict_proba(sc.transform(Xv[va]).astype(np.float32))
                    iters.append(int(m.n_iter_[0]))
                r = dict(task=t, variant=v, C=C, w=w, **{kk: round(float(x), 4) for kk, x in scores(meta, oof).items()},
                         n_called_0=int((oof.argmax(1) == 0).sum()), max_iter_used=max(iters), minutes=round((time.time() - t0) / 60, 1))
                rows.append(r)
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f"[cv] {t} {v} C={C:g} w={w}: class-0 F1 patch {r['patch_f1_0']:.3f} WSI {r['wsi_f1_0']:.3f} | "
                      f"macro {r['patch_macro']:.3f}/{r['wsi_macro']:.3f} | iters {max(iters)} ({r['minutes']} min)", flush=True)
            del Xv
    print("CV DONE", flush=True)


def choose(cv_csv: Path | str | None = None):
    """The selected (C, w) per (task, variant), from a CV table (default: this run's, else the reference)."""
    path = Path(cv_csv) if cv_csv else (OUT if OUT.is_file() else REFERENCE / "lr_tuning_cv.csv")
    cvr = pd.read_csv(path)
    pick = {}
    for (t, v), g in cvr.groupby(["task", "variant"]):
        base = g[(g.C == 1.0) & (g.w == 1)].iloc[0]
        ok = g[g.patch_macro >= base.patch_macro - 1e-9]
        g2 = ok.sort_values(["patch_f1_0", "wsi_f1_0", "patch_macro"], ascending=False)
        pick[(t, v)] = dict(g2.iloc[0], baseline_patch_macro=float(base.patch_macro), baseline_patch_f1_0=float(base.patch_f1_0))
    return pick


def final(res: Path | str | None = None, cv_csv: Path | str | None = None, tasks=None, one=False):
    """Refit each chosen setting on all train and score the rat test split, into ``res``."""
    import vmlevaluator.train.sklearn_trainer as st
    from vmlevaluator.config import load_experiment
    from vmlevaluator.evaluate.runner import evaluate_experiment
    res = Path(res) if res is not None else RES
    orig = st.build_estimator
    chosen = {}
    try:
        for (t, v), r in choose(cv_csv).items():
            if tasks and t not in tasks:
                continue
            k = 3 if t == "multiclass" else 2

            def build(cfg, num_classes, n_features, _w=float(r["w"]), _k=k):
                est = orig(cfg, num_classes, n_features)
                if str(cfg.family).upper() == "LR":
                    est.set_params(class_weight=cw(_w, list(range(_k)), _k))   # train split is class-balanced
                return est
            st.build_estimator = build
            chosen[f"{v}__{t}"] = {kk: (x.item() if hasattr(x, "item") else x) for kk, x in r.items()}
            if (res / f"{v}__{t}" / "evaluation" / "metrics.json").is_file():
                print(f"[skip] {v}__{t}", flush=True)
                continue
            cfg = load_experiment(t, v, data_root=DATA, results_root=res,
                                  overrides=[f"lr.C={float(r['C'])}", "streaming.max_in_memory_gb=8"])
            st.train_sklearn_model(cfg, verbose=True)
            out = evaluate_experiment(cfg, make_plots=False, verbose=False)
            print(f"[done] {v}__{t} C={r['C']} w={r['w']}: patch acc {out.patch.accuracy:.3f}, WSI acc {out.wsi.accuracy:.3f}", flush=True)
            if one:   # one refit per process, so memory is returned between fits
                sys.exit(3)
    finally:
        st.build_estimator = orig
    res.mkdir(parents=True, exist_ok=True)
    (res / "lr_tuned_params.json").write_text(json.dumps(chosen, indent=1, default=str), encoding="utf-8")
    print("FINAL DONE", flush=True)
    return chosen


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "final":
        final(one="one" in sys.argv[2:])
    else:
        {"extract": extract, "cv": cv}[cmd]()

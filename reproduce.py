#!/usr/bin/env python
"""One entry point for reproducing the manuscript.

Manuscript results (from the saved per-patch predictions in models/):

    python reproduce.py check                # ~2 min: all 249 numbers and Figures 1-4

The whole pipeline on synthetic data (the histology images are not distributed):

    python reproduce.py --synthetic all      # generate data, build, train every model, report
    python reproduce.py --synthetic all --epochs 3    # shorter ResNet-50 training (CPU)

With the original images and weights on disk (data/Dataset_balanced, data/human, weights in
models/), the same commands without --synthetic re-run the manuscript itself:

    python reproduce.py models [--human]     # re-run the saved models on the test data
    python reproduce.py train [--only cnn,lr,xgb,human] [--lr-cv]
    python reproduce.py compare              # compare runs/rerun/ with the manuscript
    python reproduce.py all                  # train, then compare

Nothing is ever written into data/, models/ or reference/; every output goes to runs/
(runs/synthetic/ with --synthetic). See README.md and the notebooks/ folder.
"""

from __future__ import annotations

import os
import sys

if "--synthetic" in sys.argv:                 # must be set before paths is imported
    os.environ["VML_SYNTHETIC"] = "1"

import argparse   # noqa: E402
import json       # noqa: E402
import subprocess  # noqa: E402
import time       # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import paths as P   # noqa: E402

RERUN = P.WORK / "rerun"
FIGS = ["figure1_patch_f1", "figure2_roc", "figure3_wsi_f1", "figure4_aclr_transfer"]
NOT_DISTRIBUTED = ("The histology images and trained weights are not distributed with this repository.\n"
                   "Run the same pipeline on synthetic data instead:  python reproduce.py --synthetic all")


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def environment():
    import importlib
    import platform
    print("python      ", platform.python_version())
    for m in ["torch", "torchvision", "timm", "sklearn", "xgboost", "numpy", "pandas", "matplotlib", "omegaconf"]:
        try:
            print(f"{m:12s} {importlib.import_module(m).__version__}")
        except ImportError:
            print(f"{m:12s} MISSING  (see README.md, 'Install')")
    import torch
    print("GPU         ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)")
    print("data        ", P.DATA, "" if (P.DATA / "dataset.yaml").exists() else "(not present)")
    ref = P.REFERENCE / "manuscript_numbers.json"
    if not ref.exists():
        print("MISSING:", ref)
    return ref.exists()


def have_real(weights: bool) -> bool:
    ok = (P.DATA / "dataset.yaml").exists()
    if weights:
        ok &= any(P.RESULTS.glob("*/*/weights/f1/best_model.pth"))
    if not ok:
        print(NOT_DISTRIBUTED)
    return ok


def compare_figures(out: Path) -> bool:
    from PIL import Image, ImageChops
    ok = True
    for name in FIGS:
        a = Image.open(P.MANUSCRIPT_FIGURES / f"{name}.png").convert("RGB")
        b = Image.open(out / f"{name}.png").convert("RGB")
        same = a.size == b.size and ImageChops.difference(a, b).getbbox() is None
        ok &= same
        print(f"{name:24s} {'pixel-identical to the manuscript figure' if same else 'differs from the manuscript figure'}")
    return ok


# ----------------------------------------------------------------------------- manuscript
def check(_args):
    from manuscript_figures import make_figures
    from manuscript_numbers import compare, compute
    banner("1/4  Environment and inputs")
    ok = environment()
    banner("2/4  Class-balanced dataset (Dataset_balanced): verification and counts")
    skipped = not (P.DATA / "dataset.yaml").exists()
    if skipped:
        print("SKIPPED: the histology images are not distributed. The dataset counts are in\n"
              "reference/dataset_report.json; the numbers and figures below need only the saved predictions.")
    else:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py"), "--check", "--only", "Dataset_balanced"],
                           capture_output=True, text=True)
        print(r.stdout.strip()[-1500:])
        ok &= r.returncode == 0
    banner("3/4  Every number in the manuscript, recomputed from the saved predictions")
    N = compute()
    P.WORK.mkdir(parents=True, exist_ok=True)
    (P.WORK / "manuscript_numbers.json").write_text(json.dumps(N, indent=1))
    cmp = compare(N)
    n_ok = int((cmp.status == "match").sum())
    print(f"{n_ok} of {len(cmp)} manuscript numbers reproduced to 3 decimals")
    if n_ok < len(cmp):
        print(cmp[cmp.status != "match"].to_string(index=False))
    ok &= n_ok == len(cmp)
    banner("4/4  Figures 1-4")
    out = P.WORK / "figures"
    make_figures(None, out)
    ok &= compare_figures(out)
    banner(("CHECK PASSED" if ok else "CHECK FAILED") + (" (dataset step skipped: images not distributed)" if ok and skipped else ""))
    return ok


def models(args):
    if not P.SYNTHETIC and not have_real(weights=True):
        return False
    cmd = [sys.executable, "-u", ROOT / "scripts" / "verify_models.py"]
    if args.human:
        cmd.append("--human")
    if args.models:
        cmd += ["--models", args.models]
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def train(args):
    if not P.SYNTHETIC and not have_real(weights=False):
        return False
    R = P.roots(RERUN)
    steps = args.only.split(",") if args.only else ["cnn", "lr", "xgb", "human"]
    lr_cv = args.lr_cv or P.SYNTHETIC          # synthetic data: always run the CV (it takes minutes)
    t0 = time.time()
    if "cnn" in steps:
        banner("Train ResNet-50, ResNet-50 + morph and the morphometric-only MLP")
        extra = ["--set", f"optimization.epochs={args.epochs}"] if args.epochs else []
        run([sys.executable, "-u", ROOT / "tools" / "train_eval_all.py", "--data-root", P.DATA,
             "--results-root", R["cnn"], "--models", "cnn,cnn_morph,mlp_morph", *extra], cwd=str(ROOT))
    if "lr" in steps:
        import lr_tuning
        if lr_cv:
            banner("Logistic regression: feature cache + animal-grouped 3-fold CV")
            lr_tuning.extract()
            lr_tuning.cv()
        banner("Logistic regression: refit the CV choice on all training data, score the test split")
        lr_tuning.final(R["lr"], lr_tuning.OUT if lr_cv else P.REFERENCE / "lr_tuning_cv.csv")
    if "xgb" in steps:
        banner("XGBoost, fixed settings")
        import run_xgb
        run_xgb.main(R["xgb"])
    if "human" in steps:
        banner("Zero-shot transfer of the trained ResNet-50s to the human data")
        for extra in (["--selftest"], []):
            run([sys.executable, "-u", ROOT / "scripts" / "human_eval.py", "--data-root", P.DATA,
                 "--results-root", R["cnn"], "--models", "cnn,cnn_morph", *extra])
    print(f"\ntraining finished in {(time.time() - t0) / 3600:.2f} h -> {RERUN}")
    return True


def compare_rerun(_args):
    from manuscript_figures import make_figures
    from manuscript_numbers import compare, compute
    banner(f"Re-run in {RERUN} versus the manuscript")
    N = compute(RERUN)
    (P.WORK / "rerun_numbers.json").write_text(json.dumps(N, indent=1))
    cmp = compare(N)
    cmp.to_csv(P.WORK / "rerun_vs_manuscript.csv", index=False)
    cmp["model"] = cmp.run.str.split("__").str[0]
    print(cmp.groupby(["part", "model"]).status.value_counts().unstack(fill_value=0).to_string())
    n_ok = int((cmp.status == "match").sum())
    print(f"\n{n_ok} of {len(cmp)} manuscript numbers reproduced to 3 decimals "
          f"(row by row: {P.WORK / 'rerun_vs_manuscript.csv'})")
    if len(N["rat"]) == 21 and len(N["human"]) == 6:
        out = P.WORK / "figures_rerun"
        make_figures(N, out, RERUN)
        compare_figures(out)
    return n_ok == len(cmp)


# ----------------------------------------------------------------------------- synthetic
def synthetic_data(args):
    banner("1/5  Synthetic data: generate (scripts/make_synthetic_data.py) and balance (scripts/build_dataset.py)")
    run([sys.executable, ROOT / "scripts" / "make_synthetic_data.py", "--out", P.DATASETS])
    run([sys.executable, ROOT / "scripts" / "build_dataset.py", "--only", "Dataset_balanced"])
    return True


def synthetic_report(_args):
    from manuscript_figures import make_figures, summary_table
    from manuscript_numbers import compute
    banner("4/5  Numbers and Figures 1-4 from the synthetic run")
    N = compute(RERUN)
    (P.WORK / "synthetic_numbers.json").write_text(json.dumps(N, indent=1))
    table = summary_table(N, RERUN)
    table.to_csv(P.WORK / "synthetic_summary.csv")
    print(table.drop(columns="source", errors="ignore").to_string())
    complete = len(N["rat"]) == 21 and len(N["human"]) == 6
    figs = make_figures(N, P.WORK / "figures", RERUN) if complete else []
    print(f"\n{len(N['rat'])} of 21 rat runs and {len(N['human'])} of 6 human runs scored; "
          f"{len(figs) and 4} of 4 figures drawn -> {P.WORK / 'figures'}")
    banner("5/5  Reload every saved synthetic model and check it gives its saved predictions")
    verified = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "verify_models.py"), "--human"]).returncode == 0
    ok = complete and bool(figs) and verified
    banner("SYNTHETIC PIPELINE COMPLETE" if ok else "SYNTHETIC PIPELINE INCOMPLETE")
    print("The scores above describe synthetic images only; they say nothing about the manuscript.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "data", "models", "train", "compare", "report", "all"])
    ap.add_argument("--synthetic", action="store_true", help="run on the synthetic dataset (data_synthetic/, runs/synthetic/)")
    ap.add_argument("--only", default=None, help="train: comma list of cnn, lr, xgb, human")
    ap.add_argument("--lr-cv", action="store_true", help="train: re-run the LR cross-validation (always on with --synthetic)")
    ap.add_argument("--epochs", type=int, default=None, help="train: cap ResNet-50 / MLP epochs (default: the config's 100, early stopping)")
    ap.add_argument("--human", action="store_true", help="models: also re-score the human comparisons")
    ap.add_argument("--models", default=None, help="models: comma list (default all 21)")
    args = ap.parse_args()

    if args.synthetic:
        commands = {"data": synthetic_data, "train": train, "models": models, "report": synthetic_report}
        if args.command == "all":
            ok = synthetic_data(args)
            if ok:
                banner("2-3/5  Train every model on the synthetic data, then score the synthetic human data")
                ok = train(args) and synthetic_report(args)
        elif args.command in commands:
            ok = commands[args.command](args)
        else:
            ap.error(f"'{args.command}' compares with the manuscript; it has no synthetic version")
    else:
        if args.command in ("data", "report"):
            ap.error(f"'{args.command}' is for --synthetic")
        if args.command == "all":
            ok = train(args) and compare_rerun(args)
        else:
            ok = {"check": check, "models": models, "train": train, "compare": compare_rerun}[args.command](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

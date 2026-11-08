"""``vmleval`` command line interface.

Single-experiment commands::

    vmleval info                                   # environment + data check
    vmleval tile --in slides/ --out patches/       # WSI -> patches
    vmleval split --in patches/ --out dataset/     # WSI-level train/test split
    vmleval train --task multiclass --model cnn_morph
    vmleval evaluate --task multiclass --model cnn_morph

The manuscript runs are driven by ``reproduce.py`` (which calls ``train`` /
``evaluate`` through ``tools/train_eval_all.py``) and the scripts in ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import paths
from .config import describe, load_experiment
from .constants import MODELS, TASKS


def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--quick", action="store_true", help="QUICK_RUN: few epochs / iterations")
    parser.add_argument("--data-root", default=None, help="override $VML_DATA_ROOT")
    parser.add_argument("--results-root", default=None, help="override $VML_RESULTS_ROOT")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="config override, repeatable (e.g. --set data.batch_size=8)",
    )


def _cfg(args):
    return load_experiment(
        args.task,
        args.model,
        quick=args.quick,
        data_root=Path(args.data_root) if args.data_root else None,
        results_root=Path(args.results_root) if args.results_root else None,
        overrides=args.overrides,
    )


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_info(args) -> int:
    import importlib

    print("VMLevaluator environment")
    print(f"  package root : {paths.REPO_ROOT}")
    print(f"  data root    : {paths.data_root()}"
          + ("  (bundled example subset)" if paths.using_example_data() else ""))
    print(f"  results root : {paths.results_root()}")
    print()
    for module in ("numpy", "pandas", "sklearn", "torch", "timm", "matplotlib", "pyvips"):
        try:
            m = importlib.import_module(module)
            print(f"  {module:<12} {getattr(m, '__version__', 'unknown')}")
        except ImportError:
            optional = module in ("pyvips",)
            print(f"  {module:<12} not installed" + ("  (optional: WSI tiling)" if optional else ""))

    try:
        import torch

        print(f"\n  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    print("\nTask data availability")
    for task in TASKS:
        cfg = load_experiment(task, "cnn")
        for split in ("train", "test"):
            p = Path(cfg.data[split].img_path)
            print(f"  {task:<16} {split:<5} {'ok  ' if p.is_dir() else 'MISSING'} {p}")
    return 0


def cmd_config(args) -> int:
    print(json.dumps(describe(_cfg(args)), indent=2))
    return 0


def cmd_train(args) -> int:
    from .train import train_experiment

    cfg = _cfg(args)
    result = train_experiment(cfg)
    print(json.dumps({k: str(v) for k, v in result.items() if k != "history"}, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_experiment

    cfg = _cfg(args)
    result = evaluate_experiment(cfg, model_path=args.model_path)
    print(json.dumps(result.summary_row(), indent=2, default=float))
    return 0


def cmd_tile(args) -> int:
    from .data.patches import TilingParams, tile_folder

    params = TilingParams(n_patches=args.n_patches, overlap=args.overlap)
    results = tile_folder(
        args.in_dir, args.out_dir, params,
        nodes_csv=args.nodes_csv, edges_csv=args.edges_csv, backend=args.backend,
    )
    total = sum(r["tiles"] for r in results)
    print(f"{len(results)} slides -> {total} patches in {args.out_dir}")
    return 0


def cmd_split(args) -> int:
    from .data.splits import split_by_wsi, verify_no_leakage

    plan = split_by_wsi(
        args.in_dir, args.out_dir, test_ratio=args.test_ratio, seed=args.seed, move=args.move
    )
    print(plan.summary())
    clean, overlap = verify_no_leakage(Path(args.out_dir) / "train", Path(args.out_dir) / "test")
    print("no slide overlap between splits" if clean else f"LEAKAGE: {overlap}")
    return 0 if clean else 1


# ── parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmleval",
        description="Reproducible pipeline for VML muscle-histology classification.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="environment, data roots and dataset availability")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("config", help="print a resolved experiment config")
    _add_experiment_args(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("train", help="train one model on one task")
    _add_experiment_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a trained model, writing the artefact bundle")
    _add_experiment_args(p)
    p.add_argument("--model-path", default=None, help="explicit checkpoint / joblib bundle")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("tile", help="tile whole-slide images into patches")
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--n-patches", type=int, default=400)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--nodes-csv", default=None)
    p.add_argument("--edges-csv", default=None)
    p.add_argument("--backend", choices=["auto", "pyvips", "pillow"], default="auto")
    p.set_defaults(func=cmd_tile)

    p = sub.add_parser("split", help="WSI-level train/test split of a patch folder")
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move", action="store_true")
    p.set_defaults(func=cmd_split)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

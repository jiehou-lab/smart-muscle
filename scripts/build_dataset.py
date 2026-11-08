#!/usr/bin/env python
"""Build class-balanced copies of the three rat datasets.

    python build_dataset.py            # build all three (skips a dataset already built)
    python build_dataset.py --check    # re-verify the built datasets only

For every task (control_vml, poor_very_poor, multiclass) and every split
(train, validation, test), each class is cut down to the size of that split's
smallest class. Nothing is oversampled and nothing is augmented.

* **Slides never change split.** A slide's patches all stay in the split the
  source dataset put them in; balancing only drops surplus patches. The source splits
  were checked slide- and animal-disjoint before building (see README.md).
* **Validation is balanced too.** VMLevaluator carves validation out of the
  train folder at run time (``data.validation``: 20% of animals per class,
  seeded). The same function is run here on the source train folder, so the
  slides it will hold out are known, and those slides are balanced separately
  from the remaining training slides. ``--check`` re-runs it on the built
  folder and fails if it would pick different slides.
* **Every slide keeps patches.** A class's quota is shared out evenly over its
  slides (water-filling: a slide with fewer patches than its share gives all it
  has, the rest is spread over the others). Within a slide, patches are taken
  evenly along the tile grid, so the kept ones cover the whole section rather
  than one corner.
* **Folders are severity-ordered** (folder integer == class index, most injured
  last) in all three balanced datasets, whatever order the source used; the
  ``dataset.yaml`` written beside them says so.

Sources are only read. Patches are copied, not linked.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BUILD_REPORT, DATASETS      # noqa: E402  (also puts the package on sys.path)

from omegaconf import OmegaConf                          # noqa: E402
from vmlevaluator.data.loaders import validation_slides  # noqa: E402
from vmlevaluator.data.wsi import extract_wsi_id          # noqa: E402

DATA = DATASETS
SEED = 95                      # VMLevaluator's environment.seed
VALIDATION = {"fraction": 0.2, "group_by": "animal"}
TASKS = {
    "control_vml": ["Control", "VML"],
    "poor_very_poor": ["Poor", "Very poor"],
    "multiclass": ["Perfect", "Poor", "Very poor"],
}

#: balanced name -> source folder, the source's folder->class map per task, and where
#: each task's morphometrics workbook is (relative to the source root).
SOURCES = {
    "Dataset_balanced": {
        "source": "Dataset",
        "label": "Original rat dataset (template)",
        "folder_to_class": {"control_vml": [0, 1], "poor_very_poor": [1, 0], "multiclass": [0, 2, 1]},
        "workbook": {"control_vml": "control_vml/Control_VML_dataset.xlsx",
                     "poor_very_poor": "poor_very_poor/Perfect_poor_very_poor.xlsx",
                     "multiclass": "multiclass/Perfect_poor_very_poor.xlsx"},
    },
    "patches_before_augmentation_v4": {
        "source": "patches_before_augmentation_v3",
        "label": "New rat dataset (from v3)",
        "folder_to_class": {"control_vml": [0, 1], "poor_very_poor": [0, 1], "multiclass": [0, 1, 2]},
        "workbook": {t: "morphometrics.xlsx" for t in TASKS},
    },
    "dataset_merge_v4": {
        "source": "dataset_merge_v3",
        "label": "Merged rat dataset: template + new slides (from v3)",
        "folder_to_class": {"control_vml": [0, 1], "poor_very_poor": [1, 0], "multiclass": [0, 2, 1]},
        "workbook": {t: "morphometrics.xlsx" for t in TASKS},
    },
}

_GRID = re.compile(r"__r(\d+)_c(\d+)")


def _grid_key(name: str):
    m = _GRID.search(name)
    return (int(m.group(1)), int(m.group(2)), name) if m else (10 ** 9, 10 ** 9, name)


def scan(folder: Path, f2c):
    """{class index: {slide id: [file paths in grid order]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        cls = f2c[int(sub.name)]
        for f in sub.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                out[cls][extract_wsi_id(f.name)].append(f)
    for cls in out:
        for sid in out[cls]:
            out[cls][sid].sort(key=lambda p: _grid_key(p.name))
    return out


def quotas(counts: dict, target: int) -> dict:
    """Share ``target`` over slides as evenly as their patch counts allow."""
    q = {s: 0 for s in counts}
    left = target
    open_ = sorted(counts, key=lambda s: (counts[s], s))
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for s in open_:
            if left == 0:
                break
            take = min(share, counts[s] - q[s], left)
            q[s] += take
            left -= take
            if q[s] < counts[s]:
                nxt.append(s)
        open_ = nxt
    return q


def spread(files: list, k: int, rng: random.Random) -> list:
    """``k`` files taken evenly along the grid order, with a random phase."""
    n = len(files)
    if k >= n:
        return list(files)
    step = n / k
    phase = rng.random() * step
    idx = sorted({min(n - 1, int(phase + i * step)) for i in range(k)})
    # int() can collide for step close to 1; top up from the unused files.
    if len(idx) < k:
        rest = [i for i in range(n) if i not in set(idx)]
        idx = sorted(idx + rng.sample(rest, k - len(idx)))
    return [files[i] for i in idx]


def balance(group: dict, rng: random.Random):
    """group = {cls: {slide: [files]}} -> ({cls: [files]}, per-class target, per-slide kept)"""
    totals = {c: sum(len(v) for v in group[c].values()) for c in group}
    target = min(totals.values())
    kept, per_slide = {}, {}
    for c in sorted(group):
        q = quotas({s: len(v) for s, v in group[c].items()}, target)
        kept[c] = []
        for s in sorted(group[c]):
            pick = spread(group[c][s], q[s], rng)
            kept[c].extend(pick)
            per_slide[s] = (c, len(group[c][s]), len(pick))
    return kept, target, per_slide


def held_out(train_dir: Path, f2c) -> set:
    cfg = OmegaConf.create({
        "data": {"validation": VALIDATION, "train": {"img_path": str(train_dir)},
                 "folder_to_class": list(f2c)},
        "environment": {"seed": SEED},
        "quick_run": {"enabled": False},
    })
    return set(validation_slides(cfg, verbose=False))


def build_one(name: str, spec: dict, report: dict) -> None:
    src_root, dst_root = DATA / spec["source"], DATA / name
    if dst_root.exists():
        print(f"{dst_root} exists, skipping build (delete it to rebuild)")
        return
    rng = random.Random(SEED)
    manifest = {"_comment": f"Written by build_dataset.py from {spec['source']}. "
                            "Every class is cut to the smallest class in each of train, validation "
                            "(the slides data.validation holds out) and test. Folders are "
                            "severity-ordered: folder integer == class index, most injured last.",
                "data": {"exclude_augmented": True, "online_flips": True, "validation": VALIDATION},
                "tasks": {}}
    rep = report.setdefault(name, {"source": spec["source"], "label": spec["label"], "tasks": {}})

    for task, names in TASKS.items():
        f2c = spec["folder_to_class"][task]
        tr_src, te_src = src_root / task / "train", src_root / task / "test"
        train = scan(tr_src, f2c)
        test = scan(te_src, f2c)
        held = held_out(tr_src, f2c)

        val_part = {c: {s: v for s, v in train[c].items() if s in held} for c in train}
        fit_part = {c: {s: v for s, v in train[c].items() if s not in held} for c in train}
        val_part = {c: v for c, v in val_part.items() if v}
        if len(val_part) < len(names):
            raise SystemExit(f"{name}/{task}: validation holds out slides of only "
                             f"{sorted(val_part)}; cannot balance it")

        trep = rep["tasks"].setdefault(task, {"class_names": names, "splits": {}})
        for split, part, out_split in (("train", fit_part, "train"), ("val", val_part, "train"),
                                       ("test", test, "test")):
            kept, target, per_slide = balance(part, rng)
            for c, files in kept.items():
                d = dst_root / task / out_split / str(c)
                d.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, d / f.name)
            src_n = {names[c]: sum(len(v) for v in part[c].values()) for c in sorted(part)}
            trep["splits"][split] = {
                "per_class": target, "source_per_class": src_n,
                "slides": {names[c]: len(part[c]) for c in sorted(part)},
                "slide_ids": {names[c]: sorted(part[c]) for c in sorted(part)},
                "per_slide_kept": {s: {"class": names[c], "had": had, "kept": k}
                                   for s, (c, had, k) in sorted(per_slide.items())},
            }
            print(f"{name:32s} {task:15s} {split:5s} {target:5d} per class "
                  f"(was {', '.join(f'{k} {v}' for k, v in src_n.items())}) from "
                  f"{sum(len(part[c]) for c in part)} slides")

        wb_src = src_root / spec["workbook"][task]
        wb_dst = dst_root / task / wb_src.name
        shutil.copy2(wb_src, wb_dst)
        manifest["tasks"][task] = {"data": {
            "train": {"img_path": f"{task}/train"}, "test": {"img_path": f"{task}/test"},
            "folder_to_class": list(range(len(names))), "class_names": names,
            "morphometrics_path": f"{task}/{wb_src.name}"}}

    (dst_root / "dataset.yaml").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def check(name: str, report: dict) -> list:
    """Re-verify a built dataset from disk."""
    root = DATA / name
    problems = []
    for task, names in TASKS.items():
        ident = list(range(len(names)))
        train, test = scan(root / task / "train", ident), scan(root / task / "test", ident)
        held = held_out(root / task / "train", ident)
        expect = {s for c in report[name]["tasks"][task]["splits"]["val"]["slide_ids"].values() for s in c}
        if held != expect:
            problems.append(f"{name}/{task}: validation would hold out {sorted(held)}, expected {sorted(expect)}")
        parts = {
            "train": {c: sum(len(v) for s, v in train[c].items() if s not in held) for c in train},
            "val": {c: sum(len(v) for s, v in train[c].items() if s in held) for c in train},
            "test": {c: sum(len(v) for v in test[c].values()) for c in test},
        }
        for split, cnt in parts.items():
            if len(set(cnt.values())) != 1 or len(cnt) != len(names):
                problems.append(f"{name}/{task}/{split}: not balanced {cnt}")
        tr_s = {s for c in train for s in train[c]}
        te_s = {s for c in test for s in test[c]}
        animal = lambda s: re.split(r"[_-]", s.upper())[0]
        if tr_s & te_s:
            problems.append(f"{name}/{task}: slides in train and test {sorted(tr_s & te_s)}")
        if {animal(s) for s in tr_s} & {animal(s) for s in te_s}:
            problems.append(f"{name}/{task}: animals in train and test "
                            f"{sorted({animal(s) for s in tr_s} & {animal(s) for s in te_s})}")
        dropped = [s for split in ("train", "val", "test")
                   for s, v in report[name]["tasks"][task]["splits"][split]["per_slide_kept"].items() if v["kept"] == 0]
        if dropped:
            problems.append(f"{name}/{task}: slides left with no patches {dropped}")
        print(f"{name:32s} {task:15s} train {parts['train']} val {parts['val']} test {parts['test']}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of datasets (default: all three); "
                    "the manuscript uses Dataset_balanced")
    args = ap.parse_args()
    rpath = BUILD_REPORT
    rpath.parent.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(SOURCES)
    report = json.loads(rpath.read_text()) if rpath.exists() else {}
    if not args.check:
        for name in names:
            build_one(name, SOURCES[name], report)
        rpath.write_text(json.dumps(report, indent=1), encoding="utf-8")
    problems = [p for name in names if name in report for p in check(name, report)]
    if problems:
        print("\nPROBLEMS"); [print("  ! " + p) for p in problems]; sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()

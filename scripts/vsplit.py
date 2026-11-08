"""Torch-free copy of vmlevaluator.data.loaders.validation_slides.

Importing vmlevaluator pulls in torch (~1-2 GB committed on Windows), which
fails while a sweep is training on this 14 GB machine. This reproduces the same
draw from the file names alone: slide id (extract_wsi_id), majority class per
animal group, per-class seeded shuffle, round(fraction * groups) held out.
``python vsplit.py`` checks it against the slides the real function chose for
every balanced dataset (dataset_report.json).
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def extract_wsi_id(filename: str) -> str:
    base = os.path.splitext(os.path.basename(str(filename)))[0]
    if "_" in base:
        parts = base.split("_")
        if len(parts) > 1 and parts[0].isdigit():
            base = "_".join(parts[1:])
    if "__r" in base:
        return base.split("__r")[0]
    if "__" in base:
        return base.split("__")[0]
    return base


def animal(slide_id: str) -> str:
    return re.split(r"[_-]", str(slide_id).upper())[0]


def held_out(train_dir, f2c, seed: int = 95, fraction: float = 0.2, by: str = "animal") -> set:
    root = Path(train_dir)
    slide_label = defaultdict(Counter)
    for cat in sorted(os.listdir(root)):
        if not (root / cat).is_dir():
            continue
        try:
            folder = int(cat)
        except ValueError:
            continue
        lab = int(f2c[folder]) if f2c is not None else folder
        for f in os.listdir(root / cat):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                slide_label[extract_wsi_id(f)][lab] += 1
    group_of = animal if by == "animal" else (lambda s: str(s).upper())
    groups, group_label = defaultdict(set), defaultdict(Counter)
    for sid, cnt in slide_label.items():
        g = group_of(sid)
        groups[g].add(sid)
        group_label[g].update(cnt)
    by_class = defaultdict(list)
    for g, cnt in group_label.items():
        by_class[cnt.most_common(1)[0][0]].append(g)
    rng = random.Random(int(seed))
    held = set()
    for cls in sorted(by_class):
        gs = sorted(by_class[cls])
        rng.shuffle(gs)
        if len(gs) < 2:
            continue
        n = min(len(gs) - 1, max(1, round(fraction * len(gs))))
        for g in gs[:n]:
            held |= groups[g]
    return held

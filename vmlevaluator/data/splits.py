"""Train/test partitioning.

The manuscript partitions **at the whole-slide level**: every patch from a
given WSI goes entirely to train or entirely to test, so no slide is seen on
both sides and patch-level metrics are not inflated by within-slide leakage.
:func:`split_by_wsi` is therefore the canonical splitter here.

:func:`split_patches_randomly` reproduces the older
``Data_processing/split_train_test.py`` behaviour (shuffle individual patch
files). It is kept for provenance only and warns when used, because a random
patch split leaks slide identity across the partition.
"""

from __future__ import annotations

import random
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .datasets import IMAGE_EXTENSIONS
from .wsi import extract_wsi_id

__all__ = ["SplitPlan", "plan_wsi_split", "split_by_wsi", "split_patches_randomly", "verify_no_leakage"]


@dataclass
class SplitPlan:
    """Which slides — and therefore which patches — land on each side."""

    train_slides: Dict[int, List[str]]
    test_slides: Dict[int, List[str]]
    patch_counts: Dict[str, Dict[int, int]]

    def summary(self) -> str:
        lines = ["split plan (WSI level)"]
        for split, slides in (("train", self.train_slides), ("test", self.test_slides)):
            total_slides = sum(len(v) for v in slides.values())
            total_patches = sum(self.patch_counts[split].values())
            per_class = ", ".join(f"class {k}: {len(v)}" for k, v in sorted(slides.items()))
            lines.append(
                f"  {split}: {total_slides} slides ({per_class}); {total_patches} patches"
            )
        return "\n".join(lines)


def _collect(source_dir: Path) -> Dict[int, Dict[str, List[Path]]]:
    """``{class index: {slide id: [patch paths]}}`` for a ``<root>/<class>/`` tree."""
    out: Dict[int, Dict[str, List[Path]]] = {}
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        by_slide: Dict[str, List[Path]] = defaultdict(list)
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                by_slide[extract_wsi_id(p.name)].append(p)
        if by_slide:
            out[label] = dict(by_slide)
    return out


def plan_wsi_split(
    source_dir,
    *,
    test_ratio: float = 0.3,
    seed: int = 42,
    test_slides: Optional[Sequence[str]] = None,
) -> SplitPlan:
    """Decide the split without moving any file.

    ``test_slides`` pins an explicit slide list (this is how the published
    splits are reproduced exactly); otherwise slides are sampled per class at
    ``test_ratio``, with at least one test slide per class.
    """
    grouped = _collect(Path(source_dir))
    if not grouped:
        raise RuntimeError(f"No class subfolders with images found under {source_dir}")

    rng = random.Random(seed)
    train: Dict[int, List[str]] = {}
    test: Dict[int, List[str]] = {}
    counts = {"train": defaultdict(int), "test": defaultdict(int)}
    pinned = set(test_slides or [])

    for label, by_slide in sorted(grouped.items()):
        slides = sorted(by_slide.keys())
        if pinned:
            te = [s for s in slides if s in pinned]
        else:
            shuffled = slides[:]
            rng.shuffle(shuffled)
            n_test = max(1, int(round(len(shuffled) * test_ratio)))
            te = sorted(shuffled[:n_test])
        tr = [s for s in slides if s not in te]

        train[label], test[label] = tr, te
        counts["train"][label] = sum(len(by_slide[s]) for s in tr)
        counts["test"][label] = sum(len(by_slide[s]) for s in te)

    return SplitPlan(train, test, {k: dict(v) for k, v in counts.items()})


def split_by_wsi(
    source_dir,
    out_dir,
    *,
    test_ratio: float = 0.3,
    seed: int = 42,
    test_slides: Optional[Sequence[str]] = None,
    move: bool = False,
    plan: Optional[SplitPlan] = None,
) -> SplitPlan:
    """Materialise a WSI-level split into ``out_dir/{train,test}/<class>/``."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    plan = plan or plan_wsi_split(
        source_dir, test_ratio=test_ratio, seed=seed, test_slides=test_slides
    )
    grouped = _collect(source_dir)

    for split, slides_by_class in (("train", plan.train_slides), ("test", plan.test_slides)):
        for label, slides in slides_by_class.items():
            dest = out_dir / split / str(label)
            dest.mkdir(parents=True, exist_ok=True)
            for slide in slides:
                for src in grouped[label][slide]:
                    dst = dest / src.name
                    if move:
                        shutil.move(str(src), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))
    return plan


def split_patches_randomly(
    source_dir, out_dir, *, test_ratio: float = 0.2, seed: int = 42, move: bool = False
) -> Dict[str, int]:
    """Legacy per-patch random split. Leaks slide identity — do not use for the paper."""
    warnings.warn(
        "split_patches_randomly shuffles individual patches, so patches from the same "
        "slide appear in both train and test. Use split_by_wsi for the manuscript splits.",
        UserWarning,
        stacklevel=2,
    )
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    rng = random.Random(seed)
    counts = {"train": 0, "test": 0}

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(files)
        n_test = int(len(files) * test_ratio)
        for split, chunk in (("test", files[:n_test]), ("train", files[n_test:])):
            dest = out_dir / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for src in chunk:
                (shutil.move if move else shutil.copy2)(str(src), str(dest / src.name))
                counts[split] += 1
    return counts


def verify_no_leakage(train_dir, test_dir) -> Tuple[bool, List[str]]:
    """Check that no slide contributes patches to both splits.

    Returns ``(is_clean, overlapping_slide_ids)``. Worth running in the
    notebook after any re-split — this is the invariant the whole
    patch-to-slide evaluation rests on.
    """
    def slides(root) -> set:
        out = set()
        for class_dir in Path(root).iterdir():
            if not class_dir.is_dir():
                continue
            for p in class_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    out.add(extract_wsi_id(p.name))
        return out

    overlap = sorted(slides(train_dir) & slides(test_dir))
    return (len(overlap) == 0, overlap)

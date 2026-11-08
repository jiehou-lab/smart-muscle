"""Turn a resolved config into ready-to-use datasets and loaders.

This is the single place where "which dataset does this experiment need?" is
decided, so trainers, evaluators and notebooks all build data the same way.
"""

from __future__ import annotations

import copy
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..constants import MORPHOMETRIC_ONLY, USES_MORPHOMETRICS
from ..labels import FOLDER_TO_CLASS
from .datasets import (MorphometricTableDataset, PatchDataset, build_transform,
                       scan_patch_folder)
from .morphometrics import MorphometricLookup
from .wsi import extract_wsi_id

__all__ = ["DataBundle", "build_morphometrics", "build_datasets", "build_loaders",
           "build_val_loader", "validation_slides"]


@dataclass
class DataBundle:
    """Train/test (and optional validation) datasets plus the fitted
    morphometric lookup (if any)."""

    train: object
    test: object
    morphometrics: Optional[MorphometricLookup]
    uses_morphometrics: bool
    morphometric_only: bool
    val: Optional[object] = None

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def summary(self) -> dict:
        out = {
            "train_patches": len(self.train),
            "test_patches": len(self.test),
            "train_slides": len(self.train.slide_ids),
            "test_slides": len(self.test.slide_ids),
            "train_per_class": self.train.num_samples_per_class,
            "test_per_class": self.test.num_samples_per_class,
        }
        if self.val is not None:
            out.update({
                "val_patches": len(self.val),
                "val_slides": len(self.val.slide_ids),
                "val_per_class": self.val.num_samples_per_class,
                "val_slide_ids": sorted(self.val.slide_ids),
            })
        return out


# ── validation split ─────────────────────────────────────────────────────────

AUGMENTED_TAG = "__a-"


def _animal(slide_id: str) -> str:
    """Sections of one muscle share an animal id: ``L107R_T2`` / ``ES12L_4-1``
    -> ``L107R`` / ``ES12L``. Splitting by animal keeps them together."""
    return re.split(r"[_-]", str(slide_id).upper())[0]


def validation_slides(cfg: DictConfig, verbose: bool = True) -> Set[str]:
    """Slide ids held out of the train split for checkpoint selection.

    Configured by ``data.validation`` (``null`` = no validation split, the
    published behaviour)::

        validation: {fraction: 0.2, group_by: animal}

    Groups (animals, or single slides with ``group_by: slide``) are drawn per
    class, seeded by ``environment.seed``, so every class is represented and a
    class never loses its last training group. The draw depends only on the
    slide list, so trainer, evaluator and notebooks all get the same split.
    """
    vcfg = cfg.data.get("validation")
    if not vcfg:
        return set()
    fraction = float(vcfg.get("fraction", 0.2))
    by = str(vcfg.get("group_by", "animal"))
    group_of = _animal if by == "animal" else (lambda s: str(s).upper())

    paths, labels, _ = scan_patch_folder(
        cfg.data.train.img_path, max_per_class=_max_per_class(cfg),
        label_map=_label_map(cfg))
    slide_label: Dict[str, Counter] = defaultdict(Counter)
    for p, lab in zip(paths, labels):
        slide_label[extract_wsi_id(p)][int(lab)] += 1

    groups: Dict[str, Set[str]] = defaultdict(set)
    group_label: Dict[str, Counter] = defaultdict(Counter)
    for sid, cnt in slide_label.items():
        g = group_of(sid)
        groups[g].add(sid)
        group_label[g].update(cnt)

    by_class: Dict[int, List[str]] = defaultdict(list)
    for g, cnt in group_label.items():
        by_class[cnt.most_common(1)[0][0]].append(g)

    rng = random.Random(int(cfg.environment.seed))
    held: Set[str] = set()
    for cls in sorted(by_class):
        gs = sorted(by_class[cls])
        rng.shuffle(gs)
        if len(gs) < 2:
            print(f"WARNING: class {cls} has only {len(gs)} training {by}(s); "
                  f"none held out for validation")
            continue
        n = min(len(gs) - 1, max(1, round(fraction * len(gs))))
        for g in gs[:n]:
            held |= groups[g]
    if verbose:
        print(f"validation split: {len(held)} of {len(slide_label)} train slides held out "
              f"(by {by}, fraction {fraction}): {sorted(held)}")
    return held


def _subset(ds, keep):
    """Shallow copy of a dataset restricted to the indices where ``keep`` holds."""
    out = copy.copy(ds)
    idx = [i for i, p in enumerate(ds.imgs) if keep(p)]
    out.imgs = [ds.imgs[i] for i in idx]
    out.labels = [ds.labels[i] for i in idx]
    return out


def _subset_indices(ds, indices):
    """Shallow copy of a dataset holding only ``indices``, in that order."""
    out = copy.copy(ds)
    out.imgs = [ds.imgs[i] for i in indices]
    out.labels = [ds.labels[i] for i in indices]
    return out


def _split_train_val(train, held: Set[str]):
    """Split ``train`` into (train minus held slides, held slides).

    The validation part keeps original patches only -- an augmented copy is not
    new evidence and would reweight its parent -- and uses the deterministic
    transform.
    """
    tr = _subset(train, lambda p: extract_wsi_id(p) not in held)
    va = _subset(train, lambda p: extract_wsi_id(p) in held
                 and AUGMENTED_TAG not in Path(p).name)
    if isinstance(train, PatchDataset):
        va.train_mode = False
        va.transform = build_transform(False)
    va.return_paths = False
    return tr, va


def _max_per_class(cfg: DictConfig) -> Optional[int]:
    if cfg.quick_run.get("enabled", False):
        return int(cfg.quick_run.max_patches_per_class)
    return None


def build_morphometrics(
    cfg: DictConfig,
    *,
    fit_on_train: bool = True,
    norm_stats_path: Optional[Path] = None,
    exclude_slides: Optional[Set[str]] = None,
    verbose: bool = True,
) -> MorphometricLookup:
    """Load the morphometrics workbook and set its normalisation statistics.

    Statistics come from the **training** slides only (minus ``exclude_slides``,
    i.e. the validation split), then are saved beside the model so evaluation
    and inference reuse them verbatim.
    """
    if cfg.data.morphometrics_path is None:
        raise ValueError(f"Task {cfg.meta.task!r} has no morphometrics_path configured.")

    lookup = MorphometricLookup(cfg.data.morphometrics_path, verbose=verbose)

    if norm_stats_path is not None and Path(norm_stats_path).is_file():
        lookup.load_norm_stats(norm_stats_path)
        return lookup

    if fit_on_train:
        scan = PatchDataset(
            cfg.data.train.img_path,
            train_mode=False,
            max_per_class=_max_per_class(cfg),
            verbose=False,
            label_map=_label_map(cfg),
        )
        lookup.fit([s for s in scan.slide_ids if s not in (exclude_slides or set())])
    return lookup


def _label_map(cfg):
    """The task's folder-to-class permutation, from the config.

    Falls back to :mod:`vmlevaluator.labels` when a config predates the key, so
    a hand-written config can never silently skip the severity ordering.
    """
    declared = cfg.data.get("folder_to_class")
    if declared is not None:
        return [int(v) for v in declared]
    return FOLDER_TO_CLASS.get(str(cfg.meta.task))


def build_datasets(
    cfg: DictConfig,
    *,
    morphometrics: Optional[MorphometricLookup] = None,
    train_augmentation: bool = True,
    return_paths_test: bool = True,
    with_validation: bool = True,
    verbose: bool = True,
) -> DataBundle:
    """Build the train/test datasets an experiment needs.

    ``train_augmentation`` is True for torch training and False for the
    scikit-learn models, matching the published pipelines (LR/RF extract
    features deterministically, without flips). ``data.online_flips: false``
    turns the flips off even for torch training -- for a dataset whose train
    split already carries flipped copies on disk.

    When ``data.validation`` is configured and ``with_validation`` is True, the
    train split is divided by slide into ``bundle.train`` and ``bundle.val``
    (see :func:`validation_slides`); the trainer then selects checkpoints and
    stops early on ``val`` and never looks at ``test``.
    """
    model = cfg.meta.model
    uses_morph = USES_MORPHOMETRICS[model]
    morph_only = MORPHOMETRIC_ONLY.get(model, False)
    cap = _max_per_class(cfg)
    train_augmentation = bool(train_augmentation and cfg.data.get("online_flips", True))
    held = validation_slides(cfg, verbose=verbose) if with_validation else set()

    lookup = morphometrics
    if uses_morph and lookup is None:
        lookup = build_morphometrics(cfg, exclude_slides=held, verbose=verbose)

    if morph_only:
        train = MorphometricTableDataset(
            cfg.data.train.img_path, lookup, max_per_class=cap, verbose=verbose,
            label_map=_label_map(cfg),
        )
        test = MorphometricTableDataset(
            cfg.data.test.img_path,
            lookup,
            return_paths=return_paths_test,
            max_per_class=cap,
            verbose=verbose,
            label_map=_label_map(cfg),
        )
    else:
        train = PatchDataset(
            cfg.data.train.img_path,
            train_mode=train_augmentation,
            morphometrics=lookup,
            with_morphometrics=uses_morph,
            max_per_class=cap,
            verbose=verbose,
            label_map=_label_map(cfg),
        )
        test = PatchDataset(
            cfg.data.test.img_path,
            train_mode=False,
            morphometrics=lookup,
            with_morphometrics=uses_morph,
            return_paths=return_paths_test,
            max_per_class=cap,
            verbose=verbose,
            label_map=_label_map(cfg),
        )

    if cfg.data.get("exclude_augmented", False):
        # Train on original patches only, even though augmented copies
        # (``__a-<tag>``) sit beside them on disk; online flips (if enabled)
        # then supply the augmentation.
        n0 = len(train)
        train = _subset(train, lambda p: AUGMENTED_TAG not in Path(p).name)
        test = _subset(test, lambda p: AUGMENTED_TAG not in Path(p).name)
        if verbose:
            print(f"  exclude_augmented: train {n0} -> {len(train)} original patches "
                  f"{train.num_samples_per_class}")

    val = None
    if held:
        train, val = _split_train_val(train, held)
        if verbose:
            print(f"  train -> {len(train)} patches {train.num_samples_per_class}; "
                  f"val -> {len(val)} original patches {val.num_samples_per_class}")
        missing = set(range(int(cfg.data.num_classes))) - set(val.num_samples_per_class)
        if missing:
            print(f"WARNING: validation split has no patches of class(es) {sorted(missing)}")

    return DataBundle(
        train=train,
        test=test,
        morphometrics=lookup,
        uses_morphometrics=uses_morph,
        morphometric_only=morph_only,
        val=val,
    )


class ThreadedLoader:
    """A DataLoader stand-in that fetches samples on a thread pool.

    Same batches as ``DataLoader(dataset, batch_size, shuffle)`` with default
    collation, but no worker *processes*: on Windows each worker process
    re-imports torch and its CUDA libraries (~1-2 GB committed apiece), which
    exhausts the paging file on a 14 GB machine. PNG decoding and the PIL /
    torch transforms release the GIL, so threads parallelise the real cost.
    Up to ``prefetch`` batches are prepared ahead of the one being consumed.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool, threads: int,
                 prefetch: int = 2):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.threads = int(threads)
        self.prefetch = int(prefetch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from torch.utils.data import default_collate

        n = len(self.dataset)
        order = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        batches = [order[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        get = self.dataset.__getitem__
        with ThreadPoolExecutor(self.threads) as pool:
            pending = [pool.map(get, b) for b in batches[:self.prefetch]]
            for k in range(len(batches)):
                items = list(pending.pop(0))
                nxt = k + self.prefetch
                if nxt < len(batches):
                    pending.append(pool.map(get, batches[nxt]))
                yield default_collate(items)


def _make_loader(cfg: DictConfig, dataset, shuffle: bool):
    """``data.loader_threads > 0`` (with ``num_workers: 0``) selects the
    thread-pool loader; otherwise a standard DataLoader."""
    bs = int(cfg.data.batch_size)
    threads = int(cfg.data.get("loader_threads", 0) or 0)
    nw = int(cfg.data.num_workers)
    if threads > 0 and nw == 0:
        return ThreadedLoader(dataset, bs, shuffle, threads)
    # With workers, keep them alive between epochs: on Windows every respawn
    # re-imports torch in each worker.
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      persistent_workers=nw > 0, pin_memory=nw > 0)


def build_loaders(
    cfg: DictConfig, bundle: DataBundle, *, shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_loader = _make_loader(cfg, bundle.train, shuffle_train)
    test_loader = _make_loader(cfg, bundle.test, False)
    return train_loader, test_loader


def build_val_loader(cfg: DictConfig, bundle: DataBundle) -> Optional[DataLoader]:
    """Loader over ``bundle.val``, or ``None`` when there is no validation split."""
    if bundle.val is None:
        return None
    return _make_loader(cfg, bundle.val, False)

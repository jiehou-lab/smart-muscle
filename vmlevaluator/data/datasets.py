"""Patch datasets.

One class, :class:`PatchDataset`, replaces the four near-identical dataset
classes of the original code base (``CNNOnlyDataset``, ``CNNOnlyEvalDataset``,
``CNNNumericalDataset``, ``CNNNumericalEvalDataset``). Behaviour is unchanged:

* images are read from ``<root>/<class index>/*`` — folder name *is* the label;
* keys are ``"{class}_{filename stem}"`` and iterated in sorted order, so the
  sample order is deterministic and identical to the published runs;
* train transforms add horizontal + vertical flips, test transforms do not;
* every patch inherits its parent slide's morphometrics.

What each item yields is controlled by two flags:

=====================  ========================================================
``with_morphometrics``  adds the ``(2,)`` normalised morphometric tensor
``return_paths``        adds the patch file path (needed for WSI aggregation)
=====================  ========================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .morphometrics import MorphometricLookup
from .wsi import extract_wsi_id, unique_wsi_ids

__all__ = [
    "IMAGE_EXTENSIONS",
    "build_transform",
    "PatchDataset",
    "scan_patch_folder",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: ImageNet statistics — the ResNet-50 backbone is ImageNet-pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train_mode: bool, image_size: int = 224):
    """Patch transform used by every model in the study.

    Note that the original pipeline calls ``Resize(224)`` with a single int,
    which resizes the *shorter* side to 224 and preserves aspect ratio. That is
    reproduced here exactly rather than "fixed", because the published patches
    are square and any change would alter the learned representation.
    """
    steps = [transforms.Resize(image_size)]
    if train_mode:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def scan_patch_folder(
    img_path,
    max_per_class: Optional[int] = None,
    verbose: bool = False,
    label_map: Optional[Sequence[int]] = None,
) -> tuple:
    """Return ``(paths, labels, counts)`` for a ``<root>/<class>/`` layout.

    ``label_map[folder_integer] -> class index`` re-labels the folders as they
    are read, which is how the severity ordering in :mod:`vmlevaluator.labels`
    is applied. No file is moved or copied; only the integer attached to each
    patch changes.
    """
    root = Path(img_path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Patch folder not found: {root}\n"
            "Set VML_DATA_ROOT to your dataset root, or use the bundled data_example."
        )

    img_map: Dict[str, str] = {}
    key_labels: Dict[str, int] = {}
    counts: Dict[int, int] = {}

    for category in sorted(os.listdir(root)):
        category_path = root / category
        if not category_path.is_dir():
            continue
        try:
            label = int(category)
        except ValueError:
            # Non-numeric folders (e.g. "archive") are not class folders.
            continue

        files = sorted(
            f for f in os.listdir(category_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if max_per_class is not None:
            files = files[: int(max_per_class)]

        for file in files:
            key = f"{category}_{Path(file).stem}"
            if key in img_map:
                if verbose:
                    print(f"  duplicate key '{key}' — skipping {file}")
                continue
            img_map[key] = str(category_path / file)
            key_labels[key] = label
        counts[label] = len(files)

    if not img_map:
        raise RuntimeError(
            f"No images found under {root}. Expected class subfolders named 0/, 1/, ..."
        )

    ordered_keys = sorted(img_map.keys())
    paths = [img_map[k] for k in ordered_keys]
    labels = [key_labels[k] for k in ordered_keys]

    if label_map is not None:
        n = len(label_map)
        unknown = sorted({v for v in labels if not 0 <= v < n})
        if unknown:
            raise ValueError(
                f"folder label(s) {unknown} in {root} are outside the {n} classes "
                f"declared by label_map {list(label_map)}")
        labels = [int(label_map[v]) for v in labels]
        counts = {int(label_map[k]): v for k, v in counts.items()}

    return paths, labels, counts


class PatchDataset(Dataset):
    """Histology patches, optionally paired with slide-level morphometrics.

    Parameters
    ----------
    img_path
        Folder containing one subfolder per class index.
    train_mode
        Apply flip augmentation (``True``) or deterministic transforms (``False``).
    morphometrics
        A fitted :class:`~vmlevaluator.data.morphometrics.MorphometricLookup`.
        Required when ``with_morphometrics`` is True.
    with_morphometrics, return_paths
        See module docstring.
    max_per_class
        Keep only the first N files per class (QUICK_RUN); ``None`` keeps all.
    label_map
        ``label_map[folder_integer] -> class index``. Applies the severity
        ordering from :mod:`vmlevaluator.labels` as the folders are read.
        ``None`` keeps the folder integers as-is.
    """

    def __init__(
        self,
        img_path,
        *,
        train_mode: bool = True,
        morphometrics: Optional[MorphometricLookup] = None,
        with_morphometrics: bool = False,
        return_paths: bool = False,
        max_per_class: Optional[int] = None,
        transform: Any = None,
        verbose: bool = False,
        label_map: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        if with_morphometrics and morphometrics is None:
            raise ValueError("with_morphometrics=True requires a MorphometricLookup instance.")

        self.img_path = str(img_path)
        self.train_mode = bool(train_mode)
        self.with_morphometrics = bool(with_morphometrics)
        self.return_paths = bool(return_paths)
        self.morphometrics = morphometrics
        self.transform = transform or build_transform(train_mode)
        self.loader = default_loader

        self.label_map = list(label_map) if label_map is not None else None
        self.imgs, self.labels, self._counts = scan_patch_folder(
            img_path, max_per_class=max_per_class, verbose=verbose,
            label_map=label_map,
        )

        if verbose:
            print(
                f"PatchDataset({self.img_path}): {len(self.imgs)} patches, "
                f"class distribution {self.num_samples_per_class}, "
                f"{len(self.slide_ids)} slides"
            )

    # ── torch API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        path = self.imgs[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        image = self.transform(self.loader(path))

        out: List[Any] = [image]
        if self.with_morphometrics:
            feats = self.morphometrics.get_features(extract_wsi_id(path))
            out.append(torch.as_tensor(feats, dtype=torch.float32))
        out.append(label)
        if self.return_paths:
            out.append(path)
        return tuple(out)

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def num_samples(self) -> int:
        return len(self.imgs)

    @property
    def num_samples_per_class(self) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for lbl in self.labels:
            result[lbl] = result.get(lbl, 0) + 1
        return dict(sorted(result.items()))

    @property
    def slide_ids(self) -> List[str]:
        """Unique parent slide ids present in this split."""
        return unique_wsi_ids(self.imgs)

    # Backwards-compatible alias with the original code base.
    def get_all_slide_ids(self) -> List[str]:
        return self.slide_ids

    def slide_label_table(self):
        """DataFrame of ``slide_id, label, n_patches`` — useful for Table 1."""
        import pandas as pd

        rows: Dict[str, Dict[str, Any]] = {}
        for path, label in zip(self.imgs, self.labels):
            sid = extract_wsi_id(path)
            row = rows.setdefault(sid, {"slide_id": sid, "label": label, "n_patches": 0})
            row["n_patches"] += 1
        return pd.DataFrame(list(rows.values())).sort_values(["label", "slide_id"]).reset_index(
            drop=True
        )

    def class_weights(self, num_classes: int) -> torch.Tensor:
        """Inverse-frequency weights, normalised to sum to ``num_classes``.

        Identical to the weighting used by the published trainers.
        """
        counts = self.num_samples_per_class
        total = sum(counts.values())
        weights = torch.ones(num_classes, dtype=torch.float32)
        for idx in range(num_classes):
            count = counts.get(idx, 0)
            weights[idx] = total / max(1, num_classes * max(1, count))
        return weights / weights.sum() * num_classes

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PatchDataset(n={len(self)}, classes={self.num_samples_per_class}, "
            f"morphometrics={self.with_morphometrics}, train_mode={self.train_mode})"
        )


class MorphometricTableDataset(Dataset):
    """Morphometric-only dataset for the Table-2 ablation.

    Yields ``(features, label)`` (plus the patch path when ``return_paths``),
    i.e. exactly the fusion model's numerical branch input with no image.
    Patch-level granularity is preserved so patch- and WSI-level metrics stay
    comparable with the image models.
    """

    def __init__(
        self,
        img_path,
        morphometrics: MorphometricLookup,
        *,
        return_paths: bool = False,
        max_per_class: Optional[int] = None,
        verbose: bool = False,
        label_map: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.img_path = str(img_path)
        self.morphometrics = morphometrics
        self.return_paths = bool(return_paths)
        self.label_map = list(label_map) if label_map is not None else None
        self.imgs, self.labels, _ = scan_patch_folder(
            img_path, max_per_class=max_per_class, verbose=verbose,
            label_map=label_map,
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        path = self.imgs[index]
        feats = torch.as_tensor(
            self.morphometrics.get_features(extract_wsi_id(path)), dtype=torch.float32
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return (feats, label, path) if self.return_paths else (feats, label)

    @property
    def slide_ids(self) -> List[str]:
        return unique_wsi_ids(self.imgs)

    @property
    def num_samples_per_class(self) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for lbl in self.labels:
            result[lbl] = result.get(lbl, 0) + 1
        return dict(sorted(result.items()))

    def to_arrays(self):
        """Materialise ``(X, y, paths)`` — the whole split fits in memory."""
        X = np.stack(
            [self.morphometrics.get_features(extract_wsi_id(p)) for p in self.imgs]
        ).astype(np.float32)
        y = np.asarray(self.labels, dtype=np.int64)
        return X, y, list(self.imgs)

    def class_weights(self, num_classes: int) -> torch.Tensor:
        counts = self.num_samples_per_class
        total = sum(counts.values())
        weights = torch.ones(num_classes, dtype=torch.float32)
        for idx in range(num_classes):
            weights[idx] = total / max(1, num_classes * max(1, counts.get(idx, 0)))
        return weights / weights.sum() * num_classes

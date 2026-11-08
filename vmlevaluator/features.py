"""Feature extraction for the non-deep baselines.

Logistic regression and XGBoost in this study consume the *same* tensor the
CNN sees — a 3x224x224 ImageNet-normalised patch — simply flattened to a
150,528-d vector, optionally with the two z-scored morphometric values
appended. Keeping the input identical is what makes the LR/RF numbers a fair
lower-capacity reference rather than a different experiment.

Memory note: 4,000 training patches at float32 is roughly 2.4 GB. A split
too large to hold at once is read chunk by chunk with
:func:`iter_feature_chunks`; see ``train/sklearn_trainer.py`` for how LR and RF
are fitted from chunks.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

__all__ = ["extract_features", "iter_feature_chunks", "feature_dimension",
           "describe_features", "feature_bytes"]


def _unpack(batch, with_morphometrics: bool, with_paths: bool):
    """Split a dataset batch into ``(image, morph, label, paths)``."""
    items = list(batch)
    image = items.pop(0)
    morph = items.pop(0) if with_morphometrics else None
    label = items.pop(0)
    paths = list(items.pop(0)) if with_paths else None
    return image, morph, label, paths


def extract_features(
    dataset,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    return_paths: bool = False,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Flatten a :class:`~vmlevaluator.data.datasets.PatchDataset` for sklearn.

    Returns ``(X, y, paths)`` where ``X`` is
    ``[flattened pixels]`` or ``[flattened pixels | avg_csa | num_fibers]``,
    matching ``dataset.with_morphometrics``. ``threads > 0`` (with
    ``num_workers == 0``) reads patches on a thread pool, in the same order.
    """
    with_morph = bool(getattr(dataset, "with_morphometrics", False))
    with_paths = bool(getattr(dataset, "return_paths", False))

    if threads > 0 and num_workers == 0:
        from .data.loaders import ThreadedLoader

        loader = ThreadedLoader(dataset, batch_size, shuffle=False, threads=threads)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    path_list: List[str] = []
    n_seen = 0
    n_pixel = 0

    with torch.no_grad():
        for batch in loader:
            image, morph, label, paths = _unpack(batch, with_morph, with_paths)
            flat = image.reshape(image.shape[0], -1).numpy()
            n_pixel = flat.shape[1]
            if with_morph:
                flat = np.concatenate([flat, morph.numpy()], axis=1)

            feature_chunks.append(flat.astype(np.float32, copy=False))
            label_chunks.append(label.numpy())
            if return_paths and paths is not None:
                path_list.extend(paths)

            n_seen += image.shape[0]
            if max_samples is not None and n_seen >= max_samples:
                break

    X = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(label_chunks, axis=0).astype(np.int64)

    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]
        path_list = path_list[:max_samples]

    if verbose:
        extra = f" (pixels={n_pixel} + {X.shape[1] - n_pixel} morphometric)" if with_morph else ""
        print(f"features {X.shape}{extra}, labels {y.shape}")

    if return_paths:
        if not with_paths:
            raise ValueError(
                "return_paths=True needs a dataset built with return_paths=True."
            )
        return X, y, path_list
    return X, y, None


def iter_feature_chunks(
    dataset,
    chunk_size: int,
    *,
    order: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    threads: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[List[str]]]]:
    """Yield ``(X, y, paths)`` for consecutive chunks of ``dataset``.

    ``order`` is the patch order (default: as stored); each chunk holds the
    next ``chunk_size`` of it, so a shuffled ``order`` gives chunks that mix
    classes and slides. Only one chunk's features are in memory at a time.
    """
    from .data.loaders import _subset_indices

    order = list(range(len(dataset))) if order is None else list(order)
    with_paths = bool(getattr(dataset, "return_paths", False))
    for start in range(0, len(order), int(chunk_size)):
        part = _subset_indices(dataset, order[start:start + int(chunk_size)])
        X, y, paths = extract_features(part, batch_size=batch_size, threads=threads,
                                       return_paths=with_paths, verbose=False)
        yield X, y, paths


def feature_bytes(n_patches: int, image_size: int = 224, channels: int = 3,
                  morphometrics: bool = False) -> int:
    """float32 bytes needed to hold ``n_patches`` flattened feature rows."""
    return 4 * int(n_patches) * feature_dimension(image_size, channels, morphometrics)


def feature_dimension(image_size: int = 224, channels: int = 3, morphometrics: bool = False) -> int:
    return channels * image_size * image_size + (2 if morphometrics else 0)


def describe_features(X: np.ndarray, with_morphometrics: bool = False) -> dict:
    """Small summary used in the notebooks to sanity-check the feature matrix."""
    info = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "megabytes": round(X.nbytes / 1024 ** 2, 1),
        "pixel_mean": float(X[:, :-2].mean()) if with_morphometrics else float(X.mean()),
    }
    if with_morphometrics:
        info["morphometric_mean"] = X[:, -2:].mean(axis=0).round(4).tolist()
    return info

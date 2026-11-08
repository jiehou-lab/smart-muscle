"""Whole-slide identity helpers.

Patch filenames follow the tiler's convention::

    {slide_id}__r{row}_c{col}__x{x}_y{y}__w{w}_h{h}.png

Because every train/test split is made at the WSI level and every patch
inherits its parent slide's label and morphometrics, recovering ``slide_id``
from a filename is the single most load-bearing string operation in the
pipeline. This implementation is byte-for-byte equivalent to
``CNN_only/evaluate_cnn_only.extract_wsi_id_from_filename`` in the original
code base.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List

__all__ = ["extract_wsi_id", "unique_wsi_ids", "group_paths_by_wsi"]


def extract_wsi_id(filename: str) -> str:
    """Return the slide/WSI id encoded in a patch filename or path."""
    base = os.path.splitext(os.path.basename(str(filename)))[0]

    # Drop a leading class-folder prefix such as "0_" / "1_" if present.
    if "_" in base:
        parts = base.split("_")
        if len(parts) > 1 and parts[0].isdigit():
            base = "_".join(parts[1:])

    if "__r" in base:
        return base.split("__r")[0]
    if "__" in base:
        return base.split("__")[0]
    return base


def unique_wsi_ids(paths: Iterable[str]) -> List[str]:
    """Unique slide ids, in first-seen order."""
    seen = OrderedDict()
    for p in paths:
        seen.setdefault(extract_wsi_id(p), None)
    return list(seen.keys())


def group_paths_by_wsi(paths: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = OrderedDict()
    for p in paths:
        groups.setdefault(extract_wsi_id(p), []).append(p)
    return groups

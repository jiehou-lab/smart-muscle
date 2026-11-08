"""Slide-level morphometrics: mean myofiber CSA and myofiber count.

The two features that the manuscript fuses with image representations live in
an Excel sheet keyed by slide name. This module loads that sheet, matches
patch filenames back to slide rows, and applies z-score normalisation whose
statistics are estimated **from the training slides only** (no test-set
leakage).

Accepted column spellings (case-insensitive, punctuation-insensitive):

==============  ==================================================
slide id        ``Sample ID`` | ``Image_Name`` | ``Slide`` | ``Name``
myofiber count  ``# Myofibers`` | ``#myofibers`` | ``Fiber_Count``
mean CSA        ``Average CSA`` | ``Avg. CSA`` | ``Avg_CSA`` | ``CSA``
==============  ==================================================

This is a cleaned-up but behaviour-preserving port of
``cnn_with_numerical_values/csa_lookup.py``: the matching cascade (exact ->
longest prefix -> longest substring), the NaN handling and the
"missing slide becomes the training mean" fallback are unchanged, so
normalisation statistics and model inputs match the published runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..constants import MORPHOMETRIC_FEATURES

__all__ = ["load_morphometrics_table", "MorphometricLookup"]


def _clean_path(path) -> str:
    return str(path).strip().strip('"').strip("'")


def load_morphometrics_table(excel_path) -> pd.DataFrame:
    """Read the Excel sheet into columns ``slide_id, num_fibers, avg_csa``."""
    excel_path = _clean_path(excel_path)
    df = pd.read_excel(excel_path, header=0)
    df.columns = [str(c).strip() for c in df.columns]

    col_map: Dict[str, str] = {}
    for col in df.columns:
        cl = col.lower().replace(".", "").replace(" ", "_")
        if "slide_id" not in col_map:
            if (
                ("sample" in cl and "id" in cl)
                or cl in ("image_name", "image", "slide_id", "slide", "name")
                or ("image" in cl and "name" in cl)
            ):
                col_map["slide_id"] = col
        if "num_fibers" not in col_map:
            if "myofiber" in cl or "fiber" in cl:
                col_map["num_fibers"] = col
        if "avg_csa" not in col_map:
            if "csa" in cl or ("cross" in cl and "section" in cl):
                col_map["avg_csa"] = col

    missing = [k for k in ("slide_id", "num_fibers", "avg_csa") if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in {excel_path}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Expected something like: slide id -> 'Sample ID' / 'Image_Name'; "
            "fiber count -> '# Myofibers' / 'Fiber_Count'; CSA -> 'Average CSA' / 'Avg. CSA'."
        )

    out = df[[col_map["slide_id"], col_map["num_fibers"], col_map["avg_csa"]]].copy()
    out.columns = ["slide_id", "num_fibers", "avg_csa"]

    out = out[out["slide_id"].notna() & (out["slide_id"].astype(str).str.strip() != "")]
    out["slide_id"] = out["slide_id"].astype(str).str.strip()
    out["num_fibers"] = pd.to_numeric(out["num_fibers"], errors="coerce")
    out["avg_csa"] = pd.to_numeric(out["avg_csa"], errors="coerce")
    out = out[~(out["num_fibers"].isna() & out["avg_csa"].isna())]
    return out.reset_index(drop=True)


def _match_slide_id(query: str, excel_keys: Sequence[str]) -> Optional[str]:
    """Exact -> longest prefix -> longest substring, all case-insensitive."""
    q = str(query).lower()

    for k in excel_keys:
        if k.lower() == q:
            return k

    candidates = [(k, len(k)) for k in excel_keys if q.startswith(k.lower())]
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]

    candidates = [(k, len(k)) for k in excel_keys if k.lower() in q]
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]

    return None


class MorphometricLookup:
    """Maps a patch filename to normalised ``[avg_csa, num_fibers]``.

    Parameters
    ----------
    excel_path
        Path to the slide-level morphometrics workbook.
    verbose
        Print a load summary and warn about unmatched slides.

    Notes
    -----
    Call :meth:`fit` with the *training* slide ids before reading features, or
    :meth:`load_norm_stats` to reuse the statistics saved by a previous run.
    """

    FEATURE_NAMES: List[str] = list(MORPHOMETRIC_FEATURES)
    n_features: int = 2

    def __init__(self, excel_path, verbose: bool = True):
        self.excel_path = _clean_path(excel_path)
        self.verbose = verbose

        self.table = load_morphometrics_table(self.excel_path)
        self._excel_keys: List[str] = self.table["slide_id"].tolist()

        self._raw: Dict[str, Tuple[float, float]] = {}
        for _, row in self.table.iterrows():
            avg_csa = float(row["avg_csa"]) if pd.notna(row["avg_csa"]) else np.nan
            num_fibers = float(row["num_fibers"]) if pd.notna(row["num_fibers"]) else np.nan
            self._raw[row["slide_id"]] = (avg_csa, num_fibers)

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._cache: Dict[str, Optional[str]] = {}
        self._unmatched: set = set()

        if verbose:
            print(
                f"MorphometricLookup: {len(self._raw)} slides from "
                f"{os.path.basename(self.excel_path)}; features={self.FEATURE_NAMES}"
            )
            incomplete = [k for k, v in self._raw.items() if any(np.isnan(x) for x in v)]
            if incomplete:
                print(f"  slides with missing values (mean fallback applies): {incomplete}")

    # ── normalisation ────────────────────────────────────────────────────────

    def fit(self, slide_ids: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate mean/std over ``slide_ids`` (training slides only)."""
        values = []
        for sid in slide_ids:
            raw = self._resolve_raw(sid)
            if raw is not None:
                values.append(list(raw))

        if not values:
            raise ValueError(
                "No morphometric rows matched the given slide ids; cannot compute "
                "normalisation statistics. Check that the Excel sheet covers this split."
            )

        arr = np.asarray(values, dtype=np.float32)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std[std == 0] = 1.0

        self._mean, self._std = mean, std
        if self.verbose:
            print(
                f"  norm stats — avg_csa: mean {mean[0]:.2f} / std {std[0]:.2f}; "
                f"num_fibers: mean {mean[1]:.2f} / std {std[1]:.2f}"
            )
        return mean, std

    # Backwards-compatible alias with the original code base.
    compute_norm_stats = fit

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    @property
    def mean(self) -> np.ndarray:
        self._require_fitted()
        return self._mean

    @property
    def std(self) -> np.ndarray:
        self._require_fitted()
        return self._std

    def _require_fitted(self) -> None:
        if self._mean is None:
            raise RuntimeError(
                "Normalisation statistics are not set. Call fit(train_slide_ids) "
                "or load_norm_stats(path) first."
            )

    def save_norm_stats(self, path) -> Path:
        self._require_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "mean": self._mean.tolist(),
                    "std": self._std.tolist(),
                    "features": self.FEATURE_NAMES,
                    "source": self.excel_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    def load_norm_stats(self, path) -> "MorphometricLookup":
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        self._mean = np.asarray(stats["mean"], dtype=np.float32)
        self._std = np.asarray(stats["std"], dtype=np.float32)
        return self

    # ── feature access ───────────────────────────────────────────────────────

    def _resolve_raw(self, slide_id: str) -> Optional[Tuple[float, float]]:
        if slide_id in self._cache:
            key = self._cache[slide_id]
        else:
            key = _match_slide_id(slide_id, self._excel_keys)
            self._cache[slide_id] = key
            if key is None and self.verbose and slide_id not in self._unmatched:
                self._unmatched.add(slide_id)
                print(f"  no morphometric row matched slide '{slide_id}' (mean fallback)")
        return self._raw.get(key) if key is not None else None

    def get_features(self, slide_id: str) -> np.ndarray:
        """Normalised ``[avg_csa, num_fibers]`` as float32 of shape (2,).

        Unmatched slides return zeros, i.e. the training mean after z-scoring;
        individual NaNs are likewise replaced by the training mean.
        """
        self._require_fitted()
        raw = self._resolve_raw(slide_id)
        if raw is None:
            return np.zeros(2, dtype=np.float32)

        arr = np.asarray(raw, dtype=np.float32)
        for i in range(arr.shape[0]):
            if np.isnan(arr[i]):
                arr[i] = self._mean[i]
        return ((arr - self._mean) / self._std).astype(np.float32)

    def get_raw_features(self, slide_id: str) -> Optional[Tuple[float, float]]:
        """Un-normalised ``(avg_csa, num_fibers)``, or ``None`` if unmatched."""
        return self._resolve_raw(slide_id)

    def coverage(self, slide_ids: Iterable[str]) -> pd.DataFrame:
        """Per-slide audit: which ids matched, and with what values."""
        rows = []
        for sid in dict.fromkeys(slide_ids):
            raw = self._resolve_raw(sid)
            rows.append(
                {
                    "slide_id": sid,
                    "matched": raw is not None,
                    "avg_csa": np.nan if raw is None else raw[0],
                    "num_fibers": np.nan if raw is None else raw[1],
                }
            )
        return pd.DataFrame(rows)

    def get_all_slide_ids(self) -> List[str]:
        return list(self._excel_keys)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "unfitted"
        return f"MorphometricLookup({len(self._raw)} slides, {state})"

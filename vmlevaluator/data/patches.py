"""Whole-slide tiling: tissue detection, dynamic stride, quality filtering.

Port of ``Data_processing/patches_dynamic.py``. The tiling logic is unchanged —
same grid derivation, same coarse tissue mask, same black/low-signal filters,
same ``{slide}__r{row}_c{col}__x{x}_y{y}__w{w}_h{h}.png`` filenames, which the
rest of the pipeline parses to recover slide identity.

Two readers are supported:

* **pyvips** (default when installed) — required for pyramidal WSIs
  (``.svs``, ``.ndpi``, ``.mrxs``, ``.scn``, ``.bif``) and much faster on
  large ``.tif``. This is what produced the published patches.
* **Pillow** fallback — enough for ``.png``/``.jpg``/small ``.tif`` inputs, so
  the patch-generation notebook runs on the bundled example data without a
  libvips install.

Both produce byte-identical crops for the formats they share.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "WSI_EXTENSIONS",
    "TilingParams",
    "iter_wsi_files",
    "tile_slide",
    "tile_folder",
    "is_mostly_black",
    "is_low_signal",
    "compute_grid_from_count",
    "build_tissue_mask",
]

Image.MAX_IMAGE_PIXELS = None  # histology scans exceed Pillow's decompression guard

WSI_EXTENSIONS = {
    ".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn", ".bif", ".png", ".jpg", ".jpeg",
}

PYRAMIDAL_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".scn", ".bif"}

NODE_FIELDS = [
    "slide_id", "node_id", "filename", "row", "col", "x", "y", "w", "h",
    "tile_w", "tile_h", "stride_x", "stride_y", "wsi_W", "wsi_H", "overlap_pct",
]
EDGE_FIELDS = [
    "slide_id", "src_node", "dst_node", "src_row", "src_col",
    "dst_row", "dst_col", "relation",
]


# ── tiling parameters ────────────────────────────────────────────────────────

@dataclass
class TilingParams:
    """Every knob of the tiler, with the values used for the published patches."""

    n_patches: int = 400          #: approximate tiles per slide (sets tile size)
    overlap: float = 0.0          #: percent overlap between kept tiles
    min_coverage: float = 0.6     #: min fraction of non-black pixels to keep a tile
    min_mean: float = 8.0         #: reject tiles darker than this mean intensity
    min_var: float = 8.0          #: reject tiles flatter than this variance
    brightness_thresh: int = 10   #: gray level above which a pixel counts as tissue
    png_compress: int = 6
    edge_mode: str = "black"      #: black | reflect | none — how to pad edge tiles
    scan_stride_frac: float = 0.25   #: step when a candidate tile is rejected
    min_stride_frac: float = 0.5     #: stride floor, as a fraction of tile size
    mask_min_fraction: float = 0.2   #: coarse-mask tissue fraction needed to even crop
    mask_brightness_thresh: int = 20
    max_thumb_dim: int = 1024     #: longest side of the low-res tissue mask

    def to_dict(self) -> dict:
        return asdict(self)


# ── small helpers (unchanged semantics) ──────────────────────────────────────

def to_rgb_array(arr: np.ndarray) -> np.ndarray:
    """uint8 H x W x 3, whatever the input band count."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def is_mostly_black(arr: np.ndarray, min_coverage: float = 0.6, brightness_thresh: int = 20) -> bool:
    """True when the non-black pixel fraction falls below ``min_coverage``."""
    arr = to_rgb_array(arr)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    return float((gray > brightness_thresh).mean()) < float(min_coverage)


def is_low_signal(arr: np.ndarray, min_mean: float = 10.0, min_var: float = 20.0) -> bool:
    """True when a tile is too dark or too flat to carry usable texture."""
    arr = to_rgb_array(arr)
    return (float(arr.mean()) < float(min_mean)) or (float(arr.var()) < float(min_var))


def compute_grid_from_count(W: int, H: int, N: int) -> Tuple[int, int]:
    """Columns x rows whose product is ~N and whose aspect matches the slide."""
    N = max(1, int(N))
    ratio = (W / H) if H > 0 else 1.0
    n_x = int(math.ceil(math.sqrt(N * ratio)))
    n_y = int(math.ceil(N / max(1, n_x)))
    return n_x, n_y


def save_png(arr: np.ndarray, path: Path, compress_level: int = 6) -> None:
    Image.fromarray(arr, mode="RGB").save(
        path, format="PNG", optimize=True, compress_level=int(compress_level)
    )


def iter_wsi_files(root) -> Iterator[Path]:
    """Every readable slide under ``root``, recursively, in sorted order."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in WSI_EXTENSIONS:
            yield p


# ── slide readers ────────────────────────────────────────────────────────────

class _SlideReader:
    """Minimal read interface: ``width``, ``height``, ``crop``, ``thumbnail``."""

    width: int
    height: int

    def crop(self, x: int, y: int, w: int, h: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def thumbnail(self, max_dim: int) -> Tuple[np.ndarray, float, float]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


class _VipsReader(_SlideReader):
    def __init__(self, path: Path):
        import pyvips  # imported lazily so the package works without libvips

        img = pyvips.Image.new_from_file(str(path), access="random")
        try:
            if img.interpretation != "srgb":
                img = img.colourspace("srgb")
        except Exception:
            pass
        if img.bands == 1:
            img = img.bandjoin([img, img, img])
        elif img.bands > 3:
            img = img[:3]
        self._img = img
        self.width, self.height = img.width, img.height

    @staticmethod
    def _to_numpy(img) -> np.ndarray:
        mem = img.write_to_memory()
        return np.ndarray(buffer=mem, dtype=np.uint8, shape=[img.height, img.width, img.bands])

    def crop(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        t = self._img.crop(x, y, w, h)
        return to_rgb_array(self._to_numpy(t if t.bands == 3 else t[:3]))

    def thumbnail(self, max_dim: int) -> Tuple[np.ndarray, float, float]:
        W, H = self.width, self.height
        if max(W, H) <= max_dim:
            thumb = self._img
        else:
            f = max(W, H) / float(max_dim)
            thumb = self._img.shrink(f, f)
        arr = to_rgb_array(self._to_numpy(thumb))
        return arr, float(W) / thumb.width, float(H) / thumb.height


class _PillowReader(_SlideReader):
    def __init__(self, path: Path):
        if path.suffix.lower() in PYRAMIDAL_EXTENSIONS:
            raise RuntimeError(
                f"{path.name} is a pyramidal slide format; install pyvips "
                "(`pip install pyvips` plus the libvips system library) to tile it."
            )
        self._img = Image.open(path).convert("RGB")
        self.width, self.height = self._img.size

    def crop(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return np.asarray(self._img.crop((x, y, x + w, y + h)), dtype=np.uint8)

    def thumbnail(self, max_dim: int) -> Tuple[np.ndarray, float, float]:
        W, H = self.width, self.height
        if max(W, H) <= max_dim:
            thumb = self._img
        else:
            f = max(W, H) / float(max_dim)
            thumb = self._img.resize((max(1, int(W / f)), max(1, int(H / f))))
        arr = np.asarray(thumb, dtype=np.uint8)
        return arr, float(W) / thumb.size[0], float(H) / thumb.size[1]

    def close(self) -> None:
        self._img.close()


def _open_slide(path: Path, backend: str = "auto") -> _SlideReader:
    if backend in ("auto", "pyvips"):
        try:
            return _VipsReader(path)
        except ImportError:
            if backend == "pyvips":
                raise
        except Exception:
            if backend == "pyvips":
                raise
    return _PillowReader(path)


def build_tissue_mask(
    reader: _SlideReader, max_thumb_dim: int = 1024, mask_brightness_thresh: int = 20
) -> Tuple[np.ndarray, float, float]:
    """Coarse binary tissue mask plus the thumb->full-res scale factors."""
    thumb_arr, sx, sy = reader.thumbnail(max_thumb_dim)
    gray = np.dot(to_rgb_array(thumb_arr)[..., :3], [0.299, 0.587, 0.114])
    mask = (gray > float(mask_brightness_thresh)).astype(np.uint8)
    return mask, sx, sy


# ── tiling ───────────────────────────────────────────────────────────────────

def tile_slide(
    wsi_path,
    out_dir,
    params: Optional[TilingParams] = None,
    *,
    nodes_writer: Optional[csv.DictWriter] = None,
    edges_writer: Optional[csv.DictWriter] = None,
    backend: str = "auto",
    verbose: bool = True,
) -> Dict[str, int]:
    """Tile one slide into ``out_dir``; return ``{'tiles': n, 'edges': m}``.

    A tile is written only when it survives, in order:

    1. the coarse tissue mask (``mask_min_fraction``),
    2. the mostly-black test (``min_coverage``, ``brightness_thresh``),
    3. the low-signal test (``min_mean``, ``min_var``).

    Rejected candidates advance by the short ``scan_stride_frac`` step so the
    scan can find tissue inside a mostly-empty band; accepted tiles advance by
    the full stride.
    """
    params = params or TilingParams()
    wsi_path = Path(wsi_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = wsi_path.stem
    reader = _open_slide(wsi_path, backend=backend)
    W, H = reader.width, reader.height

    tissue_mask, sx, sy = build_tissue_mask(
        reader, params.max_thumb_dim, params.mask_brightness_thresh
    )
    mask_H, mask_W = tissue_mask.shape

    n_x, n_y = compute_grid_from_count(W, H, params.n_patches)
    tile_w = math.ceil(W / n_x)
    tile_h = math.ceil(H / n_y)

    base_stride_x = max(1, int(tile_w * (1.0 - params.overlap / 100.0)))
    base_stride_y = max(1, int(tile_h * (1.0 - params.overlap / 100.0)))
    stride_x = max(max(1, int(tile_w * params.min_stride_frac)), base_stride_x)
    stride_y = max(max(1, int(tile_h * params.min_stride_frac)), base_stride_y)
    scan_stride_x = max(1, int(tile_w * params.scan_stride_frac))

    if verbose:
        print(
            f"[{slide_id}] {W}x{H} -> grid {n_x}x{n_y}, tile {tile_w}x{tile_h}, "
            f"stride ({stride_x}, {stride_y})"
        )

    rc_to_node: Dict[Tuple[int, int], str] = {}
    tiles_saved = 0

    row_idx, y = 0, 0
    while y <= H - 1:
        h = min(tile_h, H - y)
        if h <= 0:
            break
        x, col_idx = 0, 0
        while x <= W - 1:
            w = min(tile_w, W - x)
            if w <= 0:
                break

            tx0 = max(0, min(int(x / sx), mask_W - 1))
            ty0 = max(0, min(int(y / sy), mask_H - 1))
            tx1 = max(tx0 + 1, min(int((x + w) / sx), mask_W))
            ty1 = max(ty0 + 1, min(int((y + h) / sy), mask_H))
            tile_mask = tissue_mask[ty0:ty1, tx0:tx1]
            tissue_fraction = float(tile_mask.mean()) if tile_mask.size else 0.0

            if tissue_fraction < params.mask_min_fraction:
                x += stride_x
                col_idx += 1
                continue

            try:
                arr = to_rgb_array(reader.crop(x, y, w, h))
            except Exception as exc:  # pragma: no cover - reader specific
                if verbose:
                    print(f"[{slide_id}] crop failed at x={x} y={y}: {exc}")
                x += scan_stride_x
                continue

            if (w != tile_w or h != tile_h) and params.edge_mode != "none":
                mode = "reflect" if params.edge_mode == "reflect" else "constant"
                kwargs = {} if mode == "reflect" else {"constant_values": 0}
                arr = np.pad(arr, ((0, tile_h - h), (0, tile_w - w), (0, 0)), mode=mode, **kwargs)

            if is_mostly_black(arr, params.min_coverage, params.brightness_thresh) or is_low_signal(
                arr, params.min_mean, params.min_var
            ):
                x += scan_stride_x
                continue

            node_id = f"{slide_id}__r{row_idx:04d}_c{col_idx:04d}"
            out_path = out_dir / f"{node_id}__x{x}_y{y}__w{w}_h{h}.png"
            save_png(arr, out_path, params.png_compress)

            if nodes_writer is not None:
                nodes_writer.writerow(
                    {
                        "slide_id": slide_id, "node_id": node_id, "filename": str(out_path),
                        "row": row_idx, "col": col_idx, "x": x, "y": y, "w": w, "h": h,
                        "tile_w": tile_w, "tile_h": tile_h,
                        "stride_x": stride_x, "stride_y": stride_y,
                        "wsi_W": W, "wsi_H": H, "overlap_pct": params.overlap,
                    }
                )
            rc_to_node[(row_idx, col_idx)] = node_id
            tiles_saved += 1

            x += stride_x
            col_idx += 1

        y += stride_y
        row_idx += 1

    edges_written = 0
    if edges_writer is not None:
        for (r, c) in rc_to_node:
            for (nr, nc) in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                a, b = rc_to_node.get((r, c)), rc_to_node.get((nr, nc))
                if not a or not b:
                    continue
                src, dst = (a, b) if a < b else (b, a)
                edges_writer.writerow(
                    {
                        "slide_id": slide_id, "src_node": src, "dst_node": dst,
                        "src_row": r, "src_col": c, "dst_row": nr, "dst_col": nc,
                        "relation": "grid_4nbr",
                    }
                )
                edges_written += 1

    reader.close()
    if verbose:
        print(f"[{slide_id}] kept {tiles_saved} tiles")
    return {"slide_id": slide_id, "tiles": tiles_saved, "edges": edges_written}


def tile_folder(
    in_dir,
    out_dir,
    params: Optional[TilingParams] = None,
    *,
    nodes_csv=None,
    edges_csv=None,
    backend: str = "auto",
    verbose: bool = True,
) -> List[Dict[str, int]]:
    """Tile every slide under ``in_dir`` into a single flat ``out_dir``.

    ``nodes_csv`` records one row per kept tile (coordinates, size, parent
    slide) and ``edges_csv`` the 4-neighbour grid adjacency — both are the
    provenance trail that lets a reviewer trace any patch back to its slide
    region.
    """
    params = params or TilingParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_fh = edges_fh = None
    nodes_writer = edges_writer = None
    try:
        if nodes_csv is not None:
            Path(nodes_csv).parent.mkdir(parents=True, exist_ok=True)
            nodes_fh = open(nodes_csv, "w", newline="", encoding="utf-8")
            nodes_writer = csv.DictWriter(nodes_fh, fieldnames=NODE_FIELDS)
            nodes_writer.writeheader()
        if edges_csv is not None:
            Path(edges_csv).parent.mkdir(parents=True, exist_ok=True)
            edges_fh = open(edges_csv, "w", newline="", encoding="utf-8")
            edges_writer = csv.DictWriter(edges_fh, fieldnames=EDGE_FIELDS)
            edges_writer.writeheader()

        results = []
        for wsi in iter_wsi_files(in_dir):
            results.append(
                tile_slide(
                    wsi, out_dir, params,
                    nodes_writer=nodes_writer, edges_writer=edges_writer,
                    backend=backend, verbose=verbose,
                )
            )
        return results
    finally:
        if nodes_fh:
            nodes_fh.close()
        if edges_fh:
            edges_fh.close()

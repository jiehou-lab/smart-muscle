"""Image preprocessing used before tiling.

Three families of operation, ported from ``Data_processing/``:

``resize_*``
    Bring patches to a fixed square size (``resize.py``). Pillow only.
``to_red_channel`` / ``sharpen_red_edges`` / ``threshold_red``
    The laminin-signal pipeline applied to human VL scans (following ACL injury)
    (``convert2red.py`` -> ``contrast.py`` -> ``thresholded.py``). These need
    OpenCV; it is imported lazily so the rest of the package works without it.

Rat VML slides were tiled directly; the red-channel chain was used to bring
the human VL images onto a comparable appearance before transfer inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "IMAGE_EXTENSIONS",
    "resize_image",
    "resize_folder",
    "to_red_channel",
    "sharpen_red_edges",
    "threshold_red",
    "run_red_channel_pipeline",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
Image.MAX_IMAGE_PIXELS = None


def _require_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "This preprocessing step needs OpenCV. Install it with "
            "`pip install opencv-python-headless`."
        ) from exc
    import cv2

    return cv2


def _iter_images(folder) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


# ── resizing ─────────────────────────────────────────────────────────────────

def resize_image(image: Image.Image, size: int, mode: str = "resize") -> Image.Image:
    """Resize to ``size`` x ``size``.

    ``mode``
        ``resize`` — direct rescale, aspect ratio not preserved (the default
        used for the published patches, which are already near-square);
        ``pad`` — scale to fit, letterbox the remainder in black;
        ``crop`` — scale to fill, then centre-crop.
    """
    im = image.convert("RGB")
    if mode == "resize":
        return im.resize((size, size), Image.BILINEAR)

    w, h = im.size
    if mode == "pad":
        scale = min(size / w, size / h)
        new = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(new, ((size - new.size[0]) // 2, (size - new.size[1]) // 2))
        return canvas
    if mode == "crop":
        scale = max(size / w, size / h)
        new = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        left = (new.size[0] - size) // 2
        top = (new.size[1] - size) // 2
        return new.crop((left, top, left + size, top + size))
    raise ValueError(f"Unknown resize mode {mode!r}; expected resize|pad|crop.")


def resize_folder(
    in_dir, out_dir, size: int = 224, mode: str = "resize", recursive: bool = True
) -> int:
    """Resize every image under ``in_dir`` into ``out_dir``, preserving structure."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    pattern = in_dir.rglob("*") if recursive else in_dir.iterdir()
    count = 0
    for src in sorted(pattern):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dst = out_dir / src.relative_to(in_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            resize_image(im, size, mode).save(dst)
        count += 1
    return count


# ── laminin / red-channel chain (human VL scans) ───────────────────────────

def to_red_channel(bgr: np.ndarray) -> np.ndarray:
    """Collapse the blue/green laminin signal into a single red channel.

    Input and output are OpenCV BGR arrays; the output has zeros in B and G,
    so downstream steps operate on one unambiguous signal channel.
    """
    signal = np.maximum(bgr[:, :, 0], bgr[:, :, 1])
    out = np.zeros_like(bgr)
    out[:, :, 2] = signal
    return out


def sharpen_red_edges(bgr: np.ndarray) -> np.ndarray:
    """Flat-field correct, CLAHE-equalise and sharpen the red laminin signal.

    Steps: large-sigma background subtraction (removes tile illumination),
    binary tissue mask with morphological cleanup, percentile normalisation,
    CLAHE, unsharp mask plus a mild Laplacian boost, and finally masking so
    background stays black.
    """
    cv2 = _require_cv2()
    red = bgr[:, :, 2].astype(np.uint8)

    bg = cv2.GaussianBlur(red, (0, 0), sigmaX=25, sigmaY=25)
    corrected = cv2.subtract(red, bg)

    _, mask = cv2.threshold(corrected, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if np.any(corrected > 0):
        p1, p99 = np.percentile(corrected[corrected > 0], (1, 99))
    else:
        p1, p99 = 0, 255
    norm = np.clip((corrected - p1) * 255.0 / (p99 - p1 + 1e-8), 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    red_clahe = clahe.apply(norm)

    blur_small = cv2.GaussianBlur(red_clahe, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(red_clahe, 1.7, blur_small, -0.7, 0)
    lap = cv2.convertScaleAbs(cv2.Laplacian(red_clahe, cv2.CV_16S, ksize=3))
    sharp = cv2.addWeighted(sharp, 1.0, lap, 0.20, 0)

    out = np.zeros_like(bgr)
    out[:, :, 2] = cv2.bitwise_and(sharp, sharp, mask=mask)
    return out


def threshold_red(bgr: np.ndarray, min_area: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Segment laminin boundaries; return ``(masked image, binary mask)``.

    Otsu's threshold is relaxed to 60% so faint stain survives, then small
    connected components below ``min_area`` pixels are discarded as debris.
    """
    cv2 = _require_cv2()
    red = bgr[:, :, 2]
    red_blur = cv2.GaussianBlur(red, (3, 3), 0)

    otsu_val, _ = cv2.threshold(red_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    relaxed = max(3, int(0.6 * otsu_val))
    _, mask = cv2.threshold(red_blur, relaxed, 255, cv2.THRESH_BINARY)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_and(bgr, bgr, mask=cleaned), cleaned


def run_red_channel_pipeline(
    in_dir,
    out_dir,
    *,
    sharpen: bool = True,
    threshold: bool = True,
    mask_dir=None,
    min_area: int = 5000,
) -> int:
    """Apply convert-to-red -> sharpen -> threshold across a folder."""
    cv2 = _require_cv2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    for src in _iter_images(in_dir):
        img = cv2.imread(str(src))
        if img is None:
            print(f"  unreadable, skipped: {src.name}")
            continue
        out = to_red_channel(img)
        if sharpen:
            out = sharpen_red_edges(out)
        if threshold:
            out, mask = threshold_red(out, min_area=min_area)
            if mask_dir is not None:
                cv2.imwrite(str(Path(mask_dir) / src.name), mask)
        cv2.imwrite(str(out_dir / src.name), out)
        count += 1
    return count

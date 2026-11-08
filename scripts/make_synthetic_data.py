#!/usr/bin/env python
"""Generate a small synthetic dataset with the layout of the real one, for running the
whole pipeline without the (unpublished) histology images.

    python scripts/make_synthetic_data.py              # -> data_synthetic/  (~2 min, ~60 MB)
    python scripts/make_synthetic_data.py --out DIR --seed 7 --scale 2

What it makes
-------------
* Synthetic laminin-stained "whole-slide images" of muscle cross-sections: myofibres as a
  warped, size-weighted Voronoi tessellation outlined in laminin (magenta-red on black,
  like the Cy5 rat sections). One severity
  value per slide controls the look: healthy tissue has large, regular fibres with thin
  borders; injured tissue has smaller, irregular fibres, thicker interstitial laminin and
  diffuse fibrotic regions.
* Each slide is tiled into 244 x 244 patches named like the real ones
  (``<slide>__r0001_c0002__x244_y488__w244_h244.png``). Patches outside the elliptical
  tissue section are dropped.
* ``Dataset/`` mirrors the raw source dataset that ``scripts/build_dataset.py`` balances: three
  tasks with slide- and animal-disjoint train / test folders, the source's folder
  numbering (Poor vs Very poor: folder 0 = Very poor; three-class: folder 1 = Very poor,
  folder 2 = Poor), class imbalance, and the morphometric workbooks with the original
  column names. Workbook values are measured from the synthetic fibres (count and mean
  area), with noise, so they carry real but imperfect information about the class.
* ``human/`` mirrors the human transfer data: three comparisons with ``labels.csv`` and
  ``human_morphometrics.xlsx``.

Nothing here comes from real data; the numbers a model reaches on it say nothing about
the manuscript. It exists so that every step can be run and inspected.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
TILE = 244
UM_PER_PX = 1.3          # nominal scale used to turn fibre areas into "um^2"

# Severity ranges per class (0 = healthy, 1 = most injured). Neighbouring classes overlap.
RAT = {
    "control_vml": {"prefix": "SC", "classes": {"Control": (0.00, 0.30), "VML": (0.65, 1.00)},
                    "folder": {"Control": 0, "VML": 1},
                    "workbook": "Control_VML_dataset.xlsx"},
    "poor_very_poor": {"prefix": "SP", "classes": {"Poor": (0.35, 0.62), "Very poor": (0.55, 0.85)},
                       "folder": {"Very poor": 0, "Poor": 1},          # source numbering
                       "workbook": "Perfect_poor_very_poor.xlsx"},
    "multiclass": {"prefix": "SM", "classes": {"Perfect": (0.00, 0.28), "Poor": (0.35, 0.62), "Very poor": (0.55, 0.85)},
                   "folder": {"Perfect": 0, "Very poor": 1, "Poor": 2},  # source numbering
                   "workbook": "Perfect_poor_very_poor.xlsx"},
}
HUMAN_GROUPS = {"Uninjured": ("Un-injured", (0.05, 0.30)),
                "1-month post-ACLR": ("1mo post-ACLR", (0.50, 0.80)),
                "4-month post-ACLR": ("4mo post-ACLR", (0.40, 0.70))}
HUMAN_COMPARISONS = {
    "uninjured_1mo": ["Uninjured", "1-month post-ACLR"],
    "uninjured_4mo": ["Uninjured", "4-month post-ACLR"],
    "1mo_4mo": ["4-month post-ACLR", "1-month post-ACLR"],
}


# ----------------------------------------------------------------------------- rendering
def _upsample(a, factor: int, H: int, W: int):
    """Smooth (bilinear) upsampling of a coarse noise field, cropped to H x W."""
    return zoom(a, factor, order=1)[:H, :W].astype(np.float32)


def render_slide(sev: float, rng: np.random.Generator, grid=(6, 6), human=False):
    """Render one section. Returns (RGB uint8 image, tissue mask, n_fibres, mean fibre area px)."""
    H, W = grid[0] * TILE, grid[1] * TILE
    # Human sections are a moderate domain shift from the rat ones (slightly larger fibres,
    # thinner borders, less fibrosis, another hue), so zero-shot transfer is hard but not void.
    d = (64 - 32 * sev) * (1.08 if human else 1.0)               # fibre spacing, px
    d *= rng.uniform(0.92, 1.08)
    size_cv = 0.10 + 0.40 * sev
    # jittered grid of fibre centres, with additive weights for unequal sizes
    ys, xs = np.mgrid[-d:H + d:d, -d:W + d:d * 0.87]
    xs = xs + (np.arange(xs.shape[0])[:, None] % 2) * d * 0.43
    pts = np.c_[ys.ravel(), xs.ravel()] + rng.normal(0, 0.18 * d, (ys.size, 2))
    wts = rng.normal(0, size_cv * d * 0.35, len(pts))

    # smooth displacement -> irregular, wavy fibre outlines (more for injured tissue)
    amp = 4 + 18 * sev
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for arr in (yy, xx):
        f = gaussian_filter(rng.normal(0, 1, (-(-H // 4), -(-W // 4))).astype(np.float32), 6)
        f = _upsample(f / (f.std() + 1e-6), 4, H, W)
        arr += amp * f
    q = np.c_[yy.ravel(), xx.ravel()]
    dist, idx = cKDTree(pts).query(q, k=6, workers=-1)
    dw = dist - wts[idx]
    dw.sort(axis=1)
    gap = (dw[:, 1] - dw[:, 0]).reshape(H, W)                       # 0 on a fibre border

    thin = 0.75 if human else 1.0
    t = (2.2 + 4.0 * sev) * thin                                       # border half-width
    plateau = (0.5 + 6.0 * sev) * thin                                 # interstitial widening
    border = np.clip(1.0 - (gap - plateau) / t, 0, 1)
    border = np.where(gap < plateau, 1.0, border) * rng.uniform(0.8, 1.0)

    # diffuse fibrotic regions in injured tissue
    field = gaussian_filter(rng.normal(0, 1, (-(-H // 8), -(-W // 8))), 3)
    field = _upsample(field / (field.std() + 1e-6), 8, H, W)
    thr = 2.6 - 2.4 * sev ** 1.5 + (0.3 if human else 0.0)
    fib = np.clip((field - thr) / 0.6, 0, 1)
    texture = gaussian_filter(rng.random((H, W)), 2.0)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-6)
    v = np.maximum(border, fib * (0.55 + 0.4 * texture))
    v = v + rng.normal(0.04 if human else 0.06, 0.03 + 0.06 * sev, v.shape)   # interior signal + noise
    v = np.clip(gaussian_filter(v, 0.9 if human else 1.1), 0, 1)

    # elliptical tissue section
    cy, cx = H / 2 + rng.normal(0, H * 0.03), W / 2 + rng.normal(0, W * 0.03)
    ay, ax = H * rng.uniform(0.48, 0.56), W * rng.uniform(0.48, 0.56)
    mask = ((np.mgrid[0:H, 0:W][0] - cy) / ay) ** 2 + ((np.mgrid[0:H, 0:W][1] - cx) / ax) ** 2 <= 1
    v = v * mask

    if human:
        rgb = np.stack([v * 255, v * 10, v * 35], -1)
    else:
        rgb = np.stack([v * 255, v * 6, v * 62], -1)
    img = np.clip(rgb, 0, 255).astype(np.uint8)

    inside = (pts[:, 0] >= 0) & (pts[:, 0] < H) & (pts[:, 1] >= 0) & (pts[:, 1] < W)
    inside &= mask[np.clip(pts[:, 0].astype(int), 0, H - 1), np.clip(pts[:, 1].astype(int), 0, W - 1)]
    n_fibres = int(inside.sum())
    mean_area = float(mask.sum()) / max(n_fibres, 1)
    return img, mask, n_fibres, mean_area


def tile(img, mask, slide: str, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    rows, cols = img.shape[0] // TILE, img.shape[1] // TILE
    for r in range(rows):
        for c in range(cols):
            y, x = r * TILE, c * TILE
            if mask[y:y + TILE, x:x + TILE].mean() < 0.6:
                continue
            name = f"{slide}__r{r:04d}_c{c:04d}__x{x}_y{y}__w{TILE}_h{TILE}.png"
            Image.fromarray(img[y:y + TILE, x:x + TILE]).save(out_dir / name)
            names.append(name)
    return names


def morph(n_fibres, mean_area_px, rng, section_scale):
    """Workbook values: whole-section fibre count and mean CSA (um^2), measured + noise."""
    count = n_fibres * section_scale * rng.normal(1, 0.12)
    csa = mean_area_px * UM_PER_PX ** 2 * 0.55 * rng.normal(1, 0.12)
    return int(round(count)), round(float(csa), 2)


# ----------------------------------------------------------------------------- datasets
def make_rat(out: Path, rng, scale: int):
    src = out / "Dataset"
    for task, spec in RAT.items():
        rows, k = [], 0
        for cname, (lo, hi) in spec["classes"].items():
            # unequal animals per class -> the raw set is imbalanced, as in the real data
            n_train = int(rng.integers(5, 8)) * scale
            n_test = int(rng.integers(2, 4)) * scale
            for split, n in (("train", n_train), ("test", n_test)):
                for _ in range(n):
                    k += 1
                    animal = f"{spec['prefix']}{k:02d}{'LR'[k % 2]}"
                    sev = rng.uniform(lo, hi)
                    n_sections = 1 + int(rng.random() < 0.3)
                    counts, areas = [], []
                    for s in range(n_sections):
                        grid = (int(rng.integers(5, 7)), int(rng.integers(5, 8)))
                        img, mask, nf, area = render_slide(sev, rng, grid)
                        tile(img, mask, f"{animal}_{s + 1}",
                             src / task / split / str(spec["folder"][cname]))
                        counts.append(nf)
                        areas.append(area)
                    nf_, csa = morph(np.mean(counts), np.mean(areas), rng, 40)
                    rows.append({"animal": animal, "cls": cname, "count": nf_, "csa": csa, "sev": sev})
        df = pd.DataFrame(rows)
        if task == "control_vml":
            wb = pd.DataFrame({"Train Set": np.where(df.index == 0, "synthetic", None),
                               "Sample ID": df.animal, "#myofibers": df["count"].astype(float),
                               "Avg. CSA": df.csa})
        else:
            wb = pd.DataFrame({"Sample ID": df.animal, "# Myofibers": df["count"], "Average CSA": df.csa,
                               "Average Rank Score": (1 + 3 * df.sev).round(2), "Rank Group": df.cls})
        wb.to_excel(src / task / spec["workbook"], index=False)
        print(f"  {task:15s} {len(df)} animals, "
              f"{sum(1 for _ in (src / task).rglob('*.png'))} patches")


def make_human(out: Path, rng, scale: int):
    root = out / "human"
    slides = []
    k = 0
    for group, (designation, (lo, hi)) in HUMAN_GROUPS.items():
        for _ in range(4 * scale):
            k += 1
            slide = f"HV{k:02d}_{'LR'[k % 2]}"
            img, mask, nf, area = render_slide(rng.uniform(lo, hi), rng, (4, 5), human=True)
            count, csa = morph(nf, area, rng, 30)
            slides.append(dict(slide=slide, group=group, designation=designation, img=img, mask=mask,
                               Fiber_Count=count, Avg_CSA=round(csa * 1.3, 2)))
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Image_Name": s["slide"], "Fiber_Count": s["Fiber_Count"], "Avg_CSA": s["Avg_CSA"],
                   "Designation": s["designation"], "Source_Image_Name": f"{s['slide']} synthetic"}
                  for s in slides]).to_excel(root / "human_morphometrics.xlsx", index=False)
    for comp, names in HUMAN_COMPARISONS.items():
        rows = []
        for ci, group in enumerate(names):
            for s in (s for s in slides if s["group"] == group):
                for f in tile(s["img"], s["mask"], s["slide"], root / comp / str(ci)):
                    rows.append({"filename": f, "slide_id": s["slide"], "class_index": ci,
                                 "class_key": group, "designation": s["designation"],
                                 "n_fibers": s["Fiber_Count"], "avg_csa": s["Avg_CSA"]})
        pd.DataFrame(rows).to_csv(root / comp / "labels.csv", index=False)
        print(f"  human {comp:13s} {len(rows)} patches")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data_synthetic"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scale", type=int, default=1, help="multiply the number of animals (default 1)")
    ap.add_argument("--force", action="store_true", help="replace an existing output folder")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        if not args.force:
            print(f"{out} exists; use --force to regenerate")
            return
        shutil.rmtree(out)
    rng = np.random.default_rng(args.seed)
    print(f"synthetic data -> {out} (seed {args.seed})")
    make_rat(out, rng, args.scale)
    make_human(out, rng, args.scale)
    (out / "README.txt").write_text(
        "Synthetic data generated by scripts/make_synthetic_data.py "
        f"(seed {args.seed}, scale {args.scale}). Not real histology.\n")
    print("done")


if __name__ == "__main__":
    main()

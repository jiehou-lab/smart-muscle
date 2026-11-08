"""Data layer: tiling, splitting, morphometrics, datasets."""

from .datasets import MorphometricTableDataset, PatchDataset, build_transform, scan_patch_folder
from .loaders import DataBundle, build_datasets, build_loaders, build_morphometrics
from .morphometrics import MorphometricLookup, load_morphometrics_table
from .patches import TilingParams, tile_folder, tile_slide
from .preprocess import resize_folder, resize_image, run_red_channel_pipeline
from .splits import SplitPlan, plan_wsi_split, split_by_wsi, verify_no_leakage
from .wsi import extract_wsi_id, group_paths_by_wsi, unique_wsi_ids

__all__ = [
    "PatchDataset",
    "MorphometricTableDataset",
    "build_transform",
    "scan_patch_folder",
    "DataBundle",
    "build_datasets",
    "build_loaders",
    "build_morphometrics",
    "MorphometricLookup",
    "load_morphometrics_table",
    "TilingParams",
    "tile_slide",
    "tile_folder",
    "resize_image",
    "resize_folder",
    "run_red_channel_pipeline",
    "SplitPlan",
    "plan_wsi_split",
    "split_by_wsi",
    "verify_no_leakage",
    "extract_wsi_id",
    "unique_wsi_ids",
    "group_paths_by_wsi",
]

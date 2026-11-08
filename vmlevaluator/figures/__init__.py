"""Manuscript figures and tables."""

from .collect import collect_from_archive, collect_from_results_root, collect_results, save_summary
from .paper_figures import (
    build_all,
    figure1_patch_f1,
    figure2_roc,
    figure3_wsi_f1,
    figure4_aclr_transfer,
    table1_datasets,
    table2_ablation,
)
from . import style

__all__ = [
    "collect_results",
    "collect_from_results_root",
    "collect_from_archive",
    "save_summary",
    "figure1_patch_f1",
    "figure2_roc",
    "figure3_wsi_f1",
    "figure4_aclr_transfer",
    "table1_datasets",
    "table2_ablation",
    "build_all",
    "style",
]

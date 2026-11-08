"""Every path the manuscript scripts use, in one place.

Everything defaults to a location inside the repository, so the package runs wherever it
is copied. Each path can still be overridden by an environment variable:

=====================  =================================  ======================================
variable               what                               default
=====================  =================================  ======================================
``VML_DATASETS``       data root (raw ``Dataset``, built  ``data``
                       ``Dataset_balanced``, ``human``)
``VML_DATA``        the class-balanced rat dataset     ``<VML_DATASETS>/Dataset_balanced``
``VML_HUMAN_ROOT``     human comparison folders           ``<VML_DATASETS>/human``
``VML_HUMAN_WORKBOOK`` human morphometrics workbook       ``<VML_HUMAN_ROOT>/human_morphometrics.xlsx``
``VML_RESULTS``     trained models + their results     ``models``
``VML_WORK``        everything a re-run writes         ``runs``
``VML_SYNTHETIC=1``    synthetic mode: data root          ``data_synthetic``, outputs in
                       ``runs/synthetic`` (models in ``runs/synthetic/rerun``)
=====================  =================================  ======================================

Trained models are split over three folders under ``VML_RESULTS``, exactly as in
the run behind the manuscript: ResNet-50 and the morphometric-only MLP in
``neural_networks``, the CV-tuned logistic regression in
``logistic_regression``, XGBoost in ``xgboost``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent             # VMLevaluator_final/scripts
ROOT = HERE.parent                                  # VMLevaluator_final
VML = ROOT                                          # the vmlevaluator package lives here too
REFERENCE = ROOT / "reference"                      # frozen inputs and the manuscript's numbers
MANUSCRIPT_FIGURES = ROOT / "figures" / "manuscript"

#: VML_SYNTHETIC=1 points every data and output path at the synthetic dataset made by
#: scripts/make_synthetic_data.py (data_synthetic/) and at runs/synthetic/, so the whole
#: pipeline runs without the unpublished histology. reference/ and models/ are never written.
SYNTHETIC = os.environ.get("VML_SYNTHETIC", "") == "1"
_DATA_HOME = ROOT / ("data_synthetic" if SYNTHETIC else "data")
_WORK_HOME = ROOT / "runs" / "synthetic" if SYNTHETIC else ROOT / "runs"

DATASETS = Path(os.environ.get("VML_DATASETS", str(_DATA_HOME)))
DATA = Path(os.environ.get("VML_DATA", str(DATASETS / "Dataset_balanced")))
HUMAN_ROOT = Path(os.environ.get("VML_HUMAN_ROOT", str(DATASETS / "human")))
HUMAN_WB = Path(os.environ.get("VML_HUMAN_WORKBOOK", str(HUMAN_ROOT / "human_morphometrics.xlsx")))
WORK = Path(os.environ.get("VML_WORK", str(_WORK_HOME)))
RESULTS = Path(os.environ.get("VML_RESULTS", str(WORK / "rerun" if SYNTHETIC else ROOT / "models")))
#: where scripts/build_dataset.py records what it built (the real build's report is frozen in reference/)
BUILD_REPORT = WORK / "dataset_report.json" if SYNTHETIC else REFERENCE / "dataset_report.json"

# ImageNet weights ResNet-50 starts from (timm resnet50, resnet50_a1_0-14fe96d1.pth) ship in
# pretrained/, so re-training needs no internet. Child processes inherit this.
_PRETRAINED = ROOT / "pretrained" / "torch"
if (_PRETRAINED / "hub" / "checkpoints").is_dir():
    os.environ.setdefault("TORCH_HOME", str(_PRETRAINED))

#: model family -> results folder name (under RESULTS)
ROOT_NAMES = {"lr": "logistic_regression", "xgb": "xgboost",
              "cnn": "neural_networks", "mlp": "neural_networks"}

TASKS = ["control_vml", "poor_very_poor", "multiclass"]
HUMAN = ["uninjured_1mo", "uninjured_4mo", "1mo_4mo"]       # the comparisons in the manuscript
MODELS = ["lr", "lr_morph", "xgb", "xgb_morph", "cnn", "cnn_morph", "mlp_morph"]


def roots(base: Path | str | None = None) -> dict:
    """{family: results folder} under ``base`` (default RESULTS)."""
    base = Path(base) if base is not None else RESULTS
    return {k: base / v for k, v in ROOT_NAMES.items()}


for p in (str(VML), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

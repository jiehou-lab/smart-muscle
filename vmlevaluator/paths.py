"""Path resolution for the reproducible package.

Nothing in ``vmlevaluator`` hard-codes an absolute path. Three roots drive
everything, each overridable by an environment variable:

================  =========================  ====================================
Root              Environment variable       Default
================  =========================  ====================================
package root      --                         directory containing ``vmlevaluator``
data root         ``VML_DATA_ROOT``          ``<package root>/data_example``
results root      ``VML_RESULTS_ROOT``       ``<package root>/results``
================  =========================  ====================================

Set ``VML_DATA_ROOT`` to the full dataset to reproduce the published numbers::

    # Windows
    set VML_DATA_ROOT=F:\\Medical_imaging\\dataset
    # Linux / macOS
    export VML_DATA_ROOT=/mnt/Medical_imaging/dataset

The bundled ``data_example`` mirrors the full dataset's folder layout exactly,
so the same configs work against either root.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "CONFIG_DIR",
    "NOTEBOOK_DIR",
    "REFERENCE_DIR",
    "data_root",
    "results_root",
    "example_data_root",
    "ensure_dir",
]

#: Directory holding the ``vmlevaluator`` python package.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: Root of the reproducible package (contains ``configs/``, ``notebooks/`` ...).
REPO_ROOT = PACKAGE_ROOT.parent

CONFIG_DIR = REPO_ROOT / "configs"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
REFERENCE_DIR = REPO_ROOT / "results_reference"
EXAMPLE_DATA_DIR = REPO_ROOT / "data_example"


def example_data_root() -> Path:
    """Path to the small bundled example dataset."""
    return EXAMPLE_DATA_DIR


def data_root() -> Path:
    """Dataset root: ``$VML_DATA_ROOT`` when set, else the bundled example data."""
    env = os.environ.get("VML_DATA_ROOT", "").strip().strip('"').strip("'")
    if env:
        return Path(env).expanduser()
    return EXAMPLE_DATA_DIR


def results_root() -> Path:
    """Where training / evaluation artefacts are written."""
    env = os.environ.get("VML_RESULTS_ROOT", "").strip().strip('"').strip("'")
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "results"


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def using_example_data() -> bool:
    """True when the active data root is the bundled example subset."""
    try:
        return data_root().resolve() == EXAMPLE_DATA_DIR.resolve()
    except OSError:
        return False

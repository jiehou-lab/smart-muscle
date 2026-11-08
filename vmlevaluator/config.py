"""Composable experiment configuration.

An experiment configuration is built by merging three YAML fragments::

    configs/base.yaml  +  configs/tasks/<task>.yaml  +  configs/models/<model>.yaml

and then resolving every dataset path against the active data root
(:func:`vmlevaluator.paths.data_root`) and every output path against the active
results root (:func:`vmlevaluator.paths.results_root`).

Typical use from a notebook::

    from vmlevaluator.config import load_experiment
    cfg = load_experiment("multiclass", "cnn_morph", quick=True)

Typical use from the CLI::

    vmleval train --task multiclass --model cnn_morph
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from . import paths
from .constants import (
    MODELS,
    TASKS,
    TASK_CLASS_NAMES,
    TASK_NUM_CLASSES,
    experiment_name,
)

__all__ = [
    "load_experiment",
    "load_config_file",
    "save_config",
    "describe",
]


def _read_yaml(path: Path) -> DictConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Config fragment not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):  # pragma: no cover - defensive
        raise TypeError(f"Config fragment is not a mapping: {path}")
    return cfg


#: Layouts a dataset root may use, in the order they are tried. The first is
#: the study's original working layout, whose folder names carry the history of
#: how the data was collected; the second is the tidy one a published release
#: uses. Supporting both means the package reads the archive it was built
#: against *and* the dataset as it is distributed, with no flag to remember.
def _layout_candidates(task: str, split: str, configured: str) -> list:
    return [configured, f"{task}/{split}", f"{task.lower()}/{split}"]


def _resolve_data_paths(cfg: DictConfig, data_root: Path) -> None:
    """Turn the task's relative dataset paths into absolute ones.

    A relative path from the task config is tried first. If it does not exist
    under this data root but the tidy ``<task>/<split>`` layout does, that is
    used instead — so pointing ``VML_DATA_ROOT`` at a published release works
    without editing any config. An absolute path is always taken as given, and
    when neither layout exists the configured path is kept so the error names
    what was actually looked for.
    """
    task = str(cfg.meta.task) if "meta" in cfg and "task" in cfg.meta else str(cfg.get("task", ""))

    for split in ("train", "test"):
        rel = cfg.data[split].img_path
        if rel is None:
            continue
        p = Path(str(rel))
        if p.is_absolute():
            cfg.data[split].img_path = str(p)
            continue
        chosen = data_root / p
        if not chosen.is_dir():
            for candidate in _layout_candidates(task, split, str(rel))[1:]:
                if (data_root / candidate).is_dir():
                    chosen = data_root / candidate
                    break
        cfg.data[split].img_path = str(chosen)

    rel = cfg.data.get("morphometrics_path")
    if rel is not None:
        p = Path(str(rel))
        chosen = p if p.is_absolute() else data_root / p
        if not p.is_absolute() and not chosen.is_file():
            # The tidy layout keeps each task's workbook beside its splits.
            for candidate in (data_root / task / p.name, data_root / p.name):
                if candidate.is_file():
                    chosen = candidate
                    break
        cfg.data.morphometrics_path = str(chosen)


#: A dataset root may describe itself in this file. It is how a dataset whose
#: folders are laid out differently from the published one -- a different
#: folder-to-class order, its own workbooks, offline augmentation -- gets read
#: correctly without editing the task configs. Shape::
#:
#:     data:                       # applies to every task
#:       online_flips: false
#:     tasks:
#:       poor_very_poor:
#:         data:
#:           folder_to_class: [0, 1]
#:           morphometrics_path: F:\...\Perfect_poor_very_poor.xlsx
DATASET_MANIFEST = "dataset.yaml"


def _dataset_overrides(data_root: Path, task: str) -> Optional[DictConfig]:
    """The ``data_root/dataset.yaml`` overrides for ``task``, if the file exists.

    A manifest that exists but does not cover ``task`` is an error: such a
    dataset's folder order cannot be assumed to match the published one, and
    guessing is exactly how labels get silently swapped.
    """
    path = Path(data_root) / DATASET_MANIFEST
    if not path.is_file():
        return None
    man = _read_yaml(path)
    tasks = man.get("tasks") or {}
    if task not in tasks:
        raise ValueError(
            f"{path} does not describe task {task!r}; add it under 'tasks:' with "
            f"its folder_to_class so the labels are not guessed.")
    out = OmegaConf.create({})
    if man.get("data") is not None:
        out = OmegaConf.merge(out, {"data": man.data})
    out = OmegaConf.merge(out, tasks[task])
    if out.get("data", {}).get("folder_to_class") is None:
        raise ValueError(f"{path}: task {task!r} must set data.folder_to_class")
    return out


def _apply_quick_run(cfg: DictConfig) -> None:
    """Shrink the run so a notebook finishes in minutes on the example data."""
    q = cfg.quick_run
    cfg.optimization.epochs = int(q.epochs)
    cfg.optimization.patience = int(q.patience)
    cfg.lr.max_iter = int(q.max_iter)
    cfg.rf.n_estimators = int(q.n_estimators)
    cfg.data.batch_size = min(int(cfg.data.batch_size), 16)


def load_experiment(
    task: str,
    model: str,
    *,
    quick: bool = False,
    data_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    output_name: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    config_dir: Optional[Path] = None,
) -> DictConfig:
    """Build a fully resolved experiment config.

    Parameters
    ----------
    task
        One of :data:`vmlevaluator.constants.TASKS`.
    model
        One of :data:`vmlevaluator.constants.MODELS`.
    quick
        Enable QUICK_RUN (few epochs / iterations). Notebooks default to ``True``.
    data_root, results_root
        Override the dataset / results roots for this config only.
    output_name
        Folder name under the results root. Defaults to ``"<model>__<task>"``.
    overrides
        Dotlist overrides, e.g. ``["data.batch_size=8", "environment.seed=0"]``.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Expected one of {TASKS}.")
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Expected one of {MODELS}.")

    cfg_dir = Path(config_dir) if config_dir is not None else paths.CONFIG_DIR

    cfg = _read_yaml(cfg_dir / "base.yaml")
    cfg = OmegaConf.merge(cfg, _read_yaml(cfg_dir / "tasks" / f"{task}.yaml"))
    cfg = OmegaConf.merge(cfg, _read_yaml(cfg_dir / "models" / f"{model}.yaml"))

    droot = Path(data_root) if data_root is not None else paths.data_root()
    ds = _dataset_overrides(droot, task)
    if ds is not None:
        cfg = OmegaConf.merge(cfg, ds)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # Defensive: task fragments own these, but keep them consistent.
    cfg.data.num_classes = int(cfg.data.get("num_classes", TASK_NUM_CLASSES[task]))
    if not cfg.data.get("class_names"):
        cfg.data.class_names = list(TASK_CLASS_NAMES[task])

    rroot = Path(results_root) if results_root is not None else paths.results_root()
    _resolve_data_paths(cfg, droot)

    cfg.output.root = str(rroot)
    cfg.output.name = output_name or cfg.output.get("name") or experiment_name(task, model)
    cfg.output.dir = str(Path(rroot) / cfg.output.name)

    cfg.quick_run.enabled = bool(quick or cfg.quick_run.get("enabled", False))
    if cfg.quick_run.enabled:
        _apply_quick_run(cfg)

    # Provenance, written next to every artefact.
    cfg.meta = {
        "task": task,
        "model": model,
        "data_root": str(droot),
        "results_root": str(rroot),
        "quick_run": bool(cfg.quick_run.enabled),
    }
    return cfg


def load_config_file(path) -> DictConfig:
    """Load a previously saved, fully resolved config (e.g. ``config.yaml``)."""
    return _read_yaml(Path(path))


def save_config(cfg: DictConfig, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, p)
    return p


def describe(cfg: DictConfig) -> Dict[str, Any]:
    """Compact dict summary, handy for notebook display."""
    return {
        "task": cfg.meta.task,
        "model": cfg.meta.model,
        "classes": list(cfg.data.class_names),
        "train_dir": cfg.data.train.img_path,
        "test_dir": cfg.data.test.img_path,
        "morphometrics": cfg.data.morphometrics_path,
        "quick_run": bool(cfg.quick_run.enabled),
        "epochs": int(cfg.optimization.epochs),
        "seed": int(cfg.environment.seed),
        "output_dir": cfg.output.dir,
    }


def clone(cfg: DictConfig) -> DictConfig:
    return copy.deepcopy(cfg)

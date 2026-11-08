"""Unified trainer for the classical baselines (logistic regression, and the
tree-ensemble pipeline that XGBoost runs through).

Covers ``lr``, ``lr_morph``, ``rf`` and ``rf_morph`` (flattened patch pixels,
with or without morphometrics). The manuscript's XGBoost models reuse the
``rf`` / ``rf_morph`` pipeline with the estimator swapped (``scripts/run_xgb.py``).

Both estimators are fitted on ``StandardScaler``-standardised features and
handle class imbalance through class weights rather than resampling.

Splits too large to hold in memory
----------------------------------
When the flattened train + test features would exceed
``streaming.max_in_memory_gb``
the trainer switches to a chunked fit that still uses **every** train patch
(augmented copies included), reading ``streaming.chunk_patches`` at a time in
a seeded shuffled order:

* the scaler is fitted with ``partial_fit`` over all chunks;
* LR becomes ``SGDClassifier(loss="log_loss")`` -- the same model, L2
  penalty ``alpha = 1 / (C * n_train)``, balanced class weights, averaged SGD
  -- trained with ``partial_fit`` for ``streaming.lr_epochs`` passes, each in a
  fresh seeded order;
* RF grows ``n_estimators / n_chunks`` trees on each chunk and joins them into
  one forest, so every patch trains some trees and none is seen twice.

Everything is seeded, so a re-run reproduces the same model. Below the budget
the original in-memory path is used unchanged.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from ..config import save_config
from ..constants import MORPHOMETRIC_FEATURES, MORPHOMETRIC_ONLY
from ..data.loaders import DataBundle, build_datasets
from ..features import extract_features, feature_bytes, iter_feature_chunks
from ..models.factory import save_sklearn_bundle
from .torch_trainer import set_seed

__all__ = ["SklearnTrainer", "train_sklearn_model", "build_estimator", "SGDLogisticRegression"]


def _sgd_classifier_cls():
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier


class SGDLogisticRegression(_sgd_classifier_cls()):
    """``SGDClassifier(loss="log_loss")`` with numerically stable probabilities.

    For more than two classes scikit-learn normalises the one-vs-rest sigmoids,
    ``p_k = s(d_k) / sum_j s(d_j)``. On 150k pixel features the decision values
    are large enough that every ``s(d_j)`` underflows to 0 and the ratio is NaN.
    This computes the same normalisation in log space (``log s(d) =
    -log(1 + exp(-d))``, then log-sum-exp), which cannot underflow. Binary
    problems are unchanged.
    """

    def predict_proba(self, X):
        if len(self.classes_) <= 2:
            return super().predict_proba(X)
        d = np.asarray(self.decision_function(X), dtype=np.float64)
        log_s = -np.logaddexp(0.0, -d)
        log_s -= log_s.max(axis=1, keepdims=True)
        p = np.exp(log_s)
        return p / p.sum(axis=1, keepdims=True)


def _class_weight(cfg: DictConfig):
    cw = cfg.lr.get("class_weight", "balanced")
    return None if cw in (None, "none", "None", "null") else str(cw)


def _forest_class_weight(cfg: DictConfig):
    cw = cfg.rf.get("class_weight", "balanced_subsample")
    return None if cw in (None, "none", "None", "null") else str(cw)


def _monotone_vector(cfg: DictConfig, num_classes: int, n_features: int) -> np.ndarray:
    """Per-feature monotonicity constraints for the two morphometric columns.

    scikit-learn's convention is +1 increasing, -1 decreasing, 0 unconstrained,
    and the constraint is on P(class 1), which is why it is defined for binary
    problems only.

    With severity-ordered labels (:mod:`vmlevaluator.labels`) class 1 is always
    the more injured class, so the biologically correct sign is -1 for every
    binary task: the probability of the more injured class must not increase as
    myofibers get larger or more numerous. The sign still comes from the task
    config rather than being hard-coded, so a task that broke that ordering
    would have to say so.
    """
    if num_classes != 2:
        raise ValueError(
            "monotone_morphometrics is defined for binary tasks only; "
            "scikit-learn's monotonic_cst does not support more than two classes. "
            "An ordinal or one-vs-rest formulation would be needed here.")
    sign = int(cfg.data.get("morphometric_monotone_sign", -1))
    if sign == 0:
        raise ValueError(
            f"task {cfg.meta.task!r} declares morphometric_monotone_sign: 0, "
            "meaning no biologically meaningful direction is defined for it.")

    # extract_features appends the morphometric columns last, so the constraint
    # vector is zeros for every pixel and `sign` for each of them.
    n_morph = len(MORPHOMETRIC_FEATURES)
    if n_features < n_morph:
        raise ValueError(
            f"expected at least {n_morph} features for a fusion model, got {n_features}")
    cst = np.zeros(n_features, dtype=np.int8)
    cst[-n_morph:] = sign
    return cst


def build_estimator(cfg: DictConfig, num_classes: int, n_features: int):
    """Construct the LogisticRegression or RandomForestClassifier for this config."""
    family = str(cfg.family).upper()
    seed = int(cfg.environment.seed)

    if family == "LR":
        # n_jobs is deliberately not passed: lbfgs ignores it, and scikit-learn
        # >= 1.8 warns that the argument has no effect.
        return LogisticRegression(
            max_iter=int(cfg.lr.max_iter),
            solver=str(cfg.lr.solver),
            C=float(cfg.lr.C),
            class_weight=_class_weight(cfg),
            random_state=seed,
            tol=1e-4,
        )

    if family == "RF":
        from sklearn.ensemble import RandomForestClassifier

        params: Dict[str, Any] = {
            "n_estimators": int(cfg.rf.n_estimators),
            "max_depth": (None if cfg.rf.get("max_depth") in (None, "null", "None")
                          else int(cfg.rf.max_depth)),
            "max_features": cfg.rf.get("max_features", "sqrt"),
            "min_samples_leaf": int(cfg.rf.min_samples_leaf),
            "min_samples_split": int(cfg.rf.min_samples_split),
            "bootstrap": bool(cfg.rf.bootstrap),
            "class_weight": _forest_class_weight(cfg),
            "random_state": seed,
            "n_jobs": int(cfg.rf.get("n_jobs", 2)),
        }

        if bool(cfg.rf.get("monotone_morphometrics", False)):
            params["monotonic_cst"] = _monotone_vector(cfg, num_classes, n_features)
            # Fit the constrained forest on one thread. Under `monotonic_cst`
            # the tree builder is *not* reproducible across threads: with an
            # identical seed and identical data, three fits at n_jobs=2 gave
            # 0.696 / 0.675 / 0.706 accuracy on poor_very_poor, while the same
            # three at n_jobs=1 gave 0.701422 every time. The unconstrained
            # forest is bit-identical either way, so this costs about forty
            # seconds on the two binary tasks and is not worth trading for a
            # model whose published number could not be reproduced.
            params["n_jobs"] = 1

        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported family {family!r} for a scikit-learn style model.")


class SklearnTrainer:
    """Feature extraction -> standardisation -> fit -> persist bundle."""

    def __init__(
        self,
        cfg: DictConfig,
        bundle: Optional[DataBundle] = None,
        *,
        verbose: bool = True,
    ):
        self.cfg = cfg
        self.verbose = verbose
        set_seed(cfg.environment.seed)

        self.output_dir = Path(cfg.output.dir)
        (self.output_dir / "weights" / "acc").mkdir(parents=True, exist_ok=True)

        # LR/RF read patches deterministically: no flip augmentation. They have
        # no epochs or checkpoints to choose between, so no validation split --
        # they fit on the whole train split.
        self.bundle = bundle or build_datasets(
            cfg, train_augmentation=False, with_validation=False, verbose=verbose
        )
        if self.bundle.morphometrics is not None and self.bundle.morphometrics.is_fitted:
            self.bundle.morphometrics.save_norm_stats(self.output_dir / "norm_stats.json")

        save_config(cfg, self.output_dir / "config.yaml")
        self.model_path = self.output_dir / "weights" / "acc" / "best_model.joblib"
        self._log_file = open(self.output_dir / "logging.txt", "w", encoding="utf-8")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def _features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        if MORPHOMETRIC_ONLY.get(self.cfg.meta.model, False):
            X, y, _ = dataset.to_arrays()
            return X, y
        X, y, _ = extract_features(
            dataset,
            batch_size=int(self.cfg.data.batch_size),
            num_workers=int(self.cfg.data.num_workers),
            verbose=self.verbose,
        )
        return X, y

    # -- chunked fit for splits too large to hold in memory -------------------

    def _needs_streaming(self) -> bool:
        if MORPHOMETRIC_ONLY.get(self.cfg.meta.model, False):
            return False
        budget = float(self.cfg.get("streaming", {}).get("max_in_memory_gb", 1e9))
        # The in-memory path holds train and test features at once.
        need = feature_bytes(len(self.bundle.train) + len(self.bundle.test),
                             morphometrics=self.bundle.uses_morphometrics) / 2**30
        if need > budget:
            self._log(f"train+test features would need {need:.1f} GB > {budget:.1f} GB "
                      f"budget: fitting chunk by chunk on all {len(self.bundle.train)} "
                      f"train patches")
            return True
        return False

    def _chunks(self, dataset, order=None):
        st = self.cfg.streaming
        return iter_feature_chunks(
            dataset, int(st.chunk_patches), order=order,
            batch_size=int(self.cfg.data.batch_size),
            threads=int(self.cfg.data.get("loader_threads", 0) or 0))

    def _train_streaming(self, class_names, num_classes) -> Dict[str, Any]:
        cfg, st = self.cfg, self.cfg.streaming
        seed = int(cfg.environment.seed)
        family = str(cfg.family).upper()
        train, test = self.bundle.train, self.bundle.test
        n = len(train)
        chunk = int(st.chunk_patches)
        classes = np.arange(num_classes)

        def order(pass_no: int):
            return np.random.default_rng(seed + pass_no).permutation(n)

        self._log("pass 0: fitting the scaler ...")
        scaler = StandardScaler()
        n_features = None
        for X, _, _ in self._chunks(train):
            scaler.partial_fit(X)
            n_features = X.shape[1]

        if family == "LR":
            counts = np.bincount(np.asarray(train.labels), minlength=num_classes)
            weights = {int(c): float(n / (num_classes * max(1, counts[c]))) for c in classes}
            model = SGDLogisticRegression(
                loss="log_loss", penalty="l2", alpha=1.0 / (float(cfg.lr.C) * n),
                class_weight=weights, average=True, random_state=seed)
            for ep in range(1, int(st.lr_epochs) + 1):
                seen = scored = correct = 0
                for X, y, _ in self._chunks(train, order(ep)):
                    Xs = scaler.transform(X).astype(np.float32, copy=False)
                    if seen:   # accuracy on each chunk *before* learning from it
                        correct += int((model.predict(Xs) == y).sum())
                        scored += len(y)
                    model.partial_fit(Xs, y, classes=classes)
                    seen += len(y)
                self._log(f"LR epoch {ep}/{int(st.lr_epochs)}: progressive train "
                          f"accuracy {correct / max(1, scored):.4f}")
        elif family == "RF":
            n_chunks = -(-n // chunk)
            total = int(cfg.rf.n_estimators)
            per = [total // n_chunks + (1 if i < total % n_chunks else 0)
                   for i in range(n_chunks)]
            model = None
            for i, (X, y, _) in enumerate(self._chunks(train, order(1))):
                if per[i] == 0:
                    continue
                if len(np.unique(y)) != num_classes:
                    raise RuntimeError(f"chunk {i} lacks a class; raise streaming.chunk_patches")
                Xs = scaler.transform(X).astype(np.float32, copy=False)
                est = build_estimator(cfg, num_classes, Xs.shape[1])
                est.set_params(n_estimators=per[i], random_state=seed + i)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    est.fit(Xs, y)
                if model is None:
                    model = est
                else:
                    model.estimators_ += est.estimators_
                self._log(f"RF chunk {i + 1}/{n_chunks}: {per[i]} trees on {len(y)} patches")
            model.n_estimators = len(model.estimators_)
            self._log(f"forest joined: {len(model.estimators_)} trees, mean depth "
                      f"{np.mean([t.get_depth() for t in model.estimators_]):.1f}")
        else:
            raise ValueError(f"Unsupported family {family!r}")

        # Train accuracy on one seeded sample chunk; test scored in chunks.
        Xs_, ys_, _ = next(iter(self._chunks(train, order(999))))
        train_acc = accuracy_score(ys_, model.predict(scaler.transform(Xs_).astype(np.float32)))
        y_test, y_pred = [], []
        for X, y, _ in self._chunks(test):
            y_pred.append(model.predict(scaler.transform(X).astype(np.float32, copy=False)))
            y_test.append(y)
        y_test, y_test_pred = np.concatenate(y_test), np.concatenate(y_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        self._log(f"train accuracy (sample of {len(ys_)}) {train_acc:.4f} | "
                  f"test accuracy {test_acc:.4f} | macro-F1 {macro_f1:.4f}")
        self._log(classification_report(y_test, y_test_pred, target_names=class_names,
                                        zero_division=0))
        return self._save(model, scaler, num_classes, class_names, n_features,
                          train_acc, test_acc, macro_f1,
                          extra={"streaming": True, "chunk_patches": chunk,
                                 "n_train_patches": int(n),
                                 "estimator": type(model).__name__})

    def train(self) -> Dict[str, Any]:
        class_names = list(self.cfg.data.class_names)
        num_classes = int(self.cfg.data.num_classes)

        if self._needs_streaming():
            return self._train_streaming(class_names, num_classes)

        self._log("extracting training features ...")
        X_train, y_train = self._features(self.bundle.train)
        self._log("extracting test features ...")
        X_test, y_test = self._features(self.bundle.test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_test_s = scaler.transform(X_test).astype(np.float32, copy=False)

        model = build_estimator(self.cfg, num_classes, X_train_s.shape[1])
        family = str(self.cfg.family).upper()

        self._log(f"fitting {family} on {X_train_s.shape} ...")
        if family == "RF":
            # No eval_set and no early stopping: a forest is not fitted
            # sequentially, so nothing about training ever consults the test
            # split.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(X_train_s, y_train)
            self._log(f"forest fitted: {len(model.estimators_)} trees, "
                      f"mean depth {np.mean([t.get_depth() for t in model.estimators_]):.1f}")
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(X_train_s, y_train)
            n_iter = getattr(model, "n_iter_", None)
            if n_iter is not None:
                reached = int(n_iter[0]) >= int(self.cfg.lr.max_iter)
                self._log(
                    f"LR stopped after {int(n_iter[0])} iterations"
                    + (" (hit max_iter — not fully converged)" if reached else " (converged)")
                )

        train_acc = accuracy_score(y_train, model.predict(X_train_s))
        y_test_pred = model.predict(X_test_s)
        test_acc = accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        self._log(f"train accuracy {train_acc:.4f} | test accuracy {test_acc:.4f} | macro-F1 {macro_f1:.4f}")
        self._log(
            classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
        )

        return self._save(model, scaler, num_classes, class_names, int(X_train_s.shape[1]),
                          train_acc, test_acc, macro_f1)

    def _save(self, model, scaler, num_classes, class_names, n_features,
              train_acc, test_acc, macro_f1, extra=None) -> Dict[str, Any]:
        save_sklearn_bundle(
            self.model_path,
            model,
            scaler,
            num_classes,
            class_names,
            metadata={
                "task": self.cfg.meta.task,
                "model": self.cfg.meta.model,
                "feature_type": (
                    "morphometrics_only"
                    if MORPHOMETRIC_ONLY.get(self.cfg.meta.model, False)
                    else ("flattened_pixels+morphometrics" if self.bundle.uses_morphometrics
                          else "flattened_pixels")
                ),
                "n_features": int(n_features),
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "test_macro_f1": float(macro_f1),
                "quick_run": bool(self.cfg.quick_run.enabled),
                "monotone_morphometrics": bool(
                    self.cfg.get("rf", {}).get("monotone_morphometrics", False)),
                "class_names": list(class_names),
                "positive_class_index": num_classes - 1,
                **(extra or {}),
            },
        )
        summary = {
            "output_dir": str(self.output_dir),
            "model_path": str(self.model_path),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(macro_f1),
            "n_features": int(n_features),
            **(extra or {}),
        }
        (self.output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self._log_file.close()
        return summary


def train_sklearn_model(cfg: DictConfig, *, verbose: bool = True) -> Dict[str, Any]:
    return SklearnTrainer(cfg, verbose=verbose).train()

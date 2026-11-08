"""XGBClassifier with the one attribute VMLevaluator's RF training branch logs
after fitting (``estimators_``). Lives in its own module so the saved joblib
bundles can be loaded from any script (``import run_xgb_shim`` first)."""

from xgboost import XGBClassifier


class XGBForestShim(XGBClassifier):
    @property
    def estimators_(self):
        # Empty: the trainer only uses it to log tree count / mean depth.
        return []

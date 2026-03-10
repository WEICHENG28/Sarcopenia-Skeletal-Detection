"""Phase 4 inference + explainability module.

Implements:
1) Load Phase 3 artifacts (3 RF models + scaler).
2) Accept frontend-like dict input and run per-action inference.
3) Late Fusion with dynamic denominator over available actions.
4) SHAP-based Top-3 feature explanation for final fused risk score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import numpy as np

from .feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "database.csv"

ACTION_NAMES = ("Sit", "Walk", "Squat")
DEFAULT_WEIGHTS: Dict[str, float] = {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}

CONTROL_COLUMNS = {
    "Label",
    "Sit",
    "Walk",
    "Squat",
    "id",
    "ID",
    "user_id",
    "UserID",
    "timestamp",
    "Timestamp",
}


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert to float: {value!r}") from exc


@dataclass
class ActionInference:
    action: str
    weight: float
    probability: float
    used: bool
    reason: str


def _load_feature_columns_from_training_data(csv_path: Path = DATA_PATH) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("database.csv has no header row")

        rows: List[Dict[str, str]] = []
        for row in reader:
            label_raw = (row.get("Label") or "").strip()
            if label_raw == "":
                continue
            try:
                label = int(float(label_raw))
            except ValueError:
                continue
            if label in (0, 1):
                rows.append(row)

    if not rows:
        raise ValueError("No usable rows from database.csv after Label filter (Label in {0,1})")

    candidate_columns = [c for c in rows[0].keys() if c not in CONTROL_COLUMNS]
    numeric_columns: List[str] = []
    for col in candidate_columns:
        try:
            for row in rows:
                _to_float((row.get(col) or "").strip())
            numeric_columns.append(col)
        except ValueError:
            continue

    if not numeric_columns:
        raise ValueError("No numeric feature columns discovered from database.csv")
    return numeric_columns


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "done"}
    return False


def _get_action_record(payload: Mapping[str, Any], action: str) -> Dict[str, Any] | None:
    actions = payload.get("actions")
    if isinstance(actions, Mapping) and action in actions and isinstance(actions[action], Mapping):
        rec = dict(actions[action])
        rec.setdefault("performed", True)
        return rec

    key = action.lower()
    single = payload.get(key)
    if isinstance(single, Mapping):
        rec = dict(single)
        rec.setdefault("performed", True)
        return rec

    return None


def _calc_shap_values(model: Any, x_scaled: np.ndarray) -> np.ndarray:
    """Return SHAP values for positive class with shape (n_samples, n_features)."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"SHAP import failed: {exc}") from exc

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_scaled)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"SHAP computation failed: {exc}") from exc

    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise RuntimeError("Unexpected SHAP list output for binary model")
        return np.array(shap_values[1], dtype=float)

    arr = np.array(shap_values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[:, :, 1]
    if arr.ndim == 2:
        return arr

    raise RuntimeError(f"Unsupported SHAP output shape: {arr.shape}")


def _safe_top3_from_fused_shap(fused_shap_row: np.ndarray, feature_columns: List[str]) -> List[Dict[str, Any]]:
    abs_values = np.abs(fused_shap_row)
    top_indices = np.argsort(abs_values)[::-1][:3]

    top3: List[Dict[str, Any]] = []
    for idx in top_indices:
        val = float(fused_shap_row[idx])
        top3.append(
            {
                "feature": feature_columns[int(idx)],
                "direction": "increase_risk" if val >= 0 else "decrease_risk",
                "impact": val,
                "abs_impact": abs(val),
            }
        )
    return top3


def _fallback_top3(model_importance_map: Dict[str, float]) -> List[Dict[str, Any]]:
    pairs = sorted(model_importance_map.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    return [
        {
            "feature": k,
            "direction": "increase_risk" if v >= 0 else "decrease_risk",
            "impact": float(v),
            "abs_impact": abs(float(v)),
        }
        for k, v in pairs
    ]


def run_inference(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Run inference from frontend-like input payload."""
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise KeyError("payload must include 'profile' dict")

    scaler = joblib.load(MODELS_DIR / "phase3_scaler.pkl")
    models = {
        "Sit": joblib.load(MODELS_DIR / "phase3_rf_sit.pkl"),
        "Walk": joblib.load(MODELS_DIR / "phase3_rf_walk.pkl"),
        "Squat": joblib.load(MODELS_DIR / "phase3_rf_squat.pkl"),
    }

    feature_columns = _load_feature_columns_from_training_data(DATA_PATH)
    per_action: Dict[str, ActionInference] = {}
    weighted_prob_sum = 0.0
    weight_sum = 0.0

    fused_shap: np.ndarray | None = None
    shap_ok = True
    shap_error = ""

    fallback_importance: Dict[str, float] = {name: 0.0 for name in feature_columns}

    for action in ACTION_NAMES:
        rec = _get_action_record(payload, action)
        if rec is None:
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, "action_not_provided")
            continue

        if not _bool_like(rec.get("performed", True)):
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, "action_not_performed")
            continue

        mot_path = rec.get("mot_path")
        trc_path = rec.get("trc_path")
        if not mot_path:
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, "missing_mot_path")
            continue

        try:
            feats = extract_features(
                profile=dict(profile),
                mot_path=str(mot_path),
                trc_path=None if trc_path is None else str(trc_path),
            )
            x_raw = np.array([[float(feats.get(col, 0.0)) for col in feature_columns]], dtype=float)
            x_scaled = scaler.transform(x_raw)
            proba = float(models[action].predict_proba(x_scaled)[0, 1])

            w = DEFAULT_WEIGHTS[action]
            weighted_prob_sum += w * proba
            weight_sum += w
            per_action[action] = ActionInference(action, w, proba, True, "ok")

            importances = getattr(models[action], "feature_importances_", None)
            if importances is not None:
                for i, col in enumerate(feature_columns):
                    fallback_importance[col] += float(w) * float(importances[i])

            if shap_ok:
                try:
                    shap_row = _calc_shap_values(models[action], x_scaled)[0]
                    if fused_shap is None:
                        fused_shap = float(w) * shap_row
                    else:
                        fused_shap = fused_shap + float(w) * shap_row
                except Exception as exc:  # pragma: no cover
                    shap_ok = False
                    shap_error = str(exc)

        except Exception as exc:
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, f"inference_failed: {exc}")
            continue

    if weight_sum <= 0:
        return {
            "ok": False,
            "message": "No usable action for inference (none performed or all failed).",
            "final_risk_probability": None,
            "used_actions": [],
            "per_action": {k: vars(v) for k, v in per_action.items()},
            "explanations_top3": [],
            "explanation_mode": "none",
        }

    final_prob = weighted_prob_sum / weight_sum
    used_actions = [a for a, r in per_action.items() if r.used]

    if shap_ok and fused_shap is not None:
        fused_shap = fused_shap / weight_sum
        top3 = _safe_top3_from_fused_shap(fused_shap, feature_columns)
        explanation_mode = "shap"
    else:
        normalized_importance = {k: (v / weight_sum) for k, v in fallback_importance.items()}
        top3 = _fallback_top3(normalized_importance)
        explanation_mode = f"fallback_feature_importance ({shap_error})" if shap_error else "fallback_feature_importance"

    return {
        "ok": True,
        "message": "inference_success",
        "final_risk_probability": float(final_prob),
        "used_actions": used_actions,
        "dynamic_weight_denominator": float(weight_sum),
        "per_action": {k: vars(v) for k, v in per_action.items()},
        "explanations_top3": top3,
        "explanation_mode": explanation_mode,
    }


"""Shift-GCN 多模態推論模組。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from .models_shiftgcn import MEDIAPIPE_NUM_NODES, ShiftGCNMultiModal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

ACTION_NAMES: Tuple[str, ...] = ("Sit", "Walk", "Squat")
DEFAULT_WEIGHTS: Dict[str, float] = {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}


@dataclass
class ActionInference:
    action: str
    weight: float
    probability: float
    used: bool
    reason: str


@dataclass
class LoadedActionModel:
    action: str
    model: ShiftGCNMultiModal
    t_dim: int
    v_dim: int
    s_dim: int


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


def _read_metadata() -> Dict[str, Any]:
    metadata_path = MODELS_DIR / "phase4_shiftgcn_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("phase4_shiftgcn_metadata.json must be a JSON object")
    return payload


def _load_action_model(action: str, t_dim: int, v_dim: int, s_dim: int, device: torch.device) -> LoadedActionModel:
    ckpt_path = MODELS_DIR / f"phase4_shiftgcn_{action.lower()}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

    ckpt_spec = checkpoint.get("input_spec") or {}
    if isinstance(ckpt_spec, Mapping):
        ckpt_t = int(ckpt_spec.get("T", t_dim))
        ckpt_v = int(ckpt_spec.get("V", v_dim))
        ckpt_s = int(ckpt_spec.get("S", s_dim))
        if ckpt_t != t_dim or ckpt_v != v_dim or ckpt_s != s_dim:
            raise ValueError(
                f"Input spec mismatch for {action}: "
                f"metadata(T={t_dim},V={v_dim},S={s_dim}) vs checkpoint(T={ckpt_t},V={ckpt_v},S={ckpt_s})"
            )

    model = ShiftGCNMultiModal(static_dim=s_dim, num_nodes=v_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return LoadedActionModel(action=action, model=model, t_dim=t_dim, v_dim=v_dim, s_dim=s_dim)


def _profile_to_static(profile: Mapping[str, Any], expected_static_cols: List[str]) -> List[float]:
    age = float(profile.get("age", 0.0))
    height = float(profile.get("height_cm", 0.0))
    weight = float(profile.get("weight_kg", 0.0))
    h_m = max(height / 100.0, 1e-6)
    bmi = float(profile.get("bmi", weight / (h_m * h_m)))

    base = {
        "static_age": age,
        "static_bmi": bmi,
        "static_height_cm": height,
        "static_weight_kg": weight,
    }
    return [float(base.get(col, 0.0)) for col in expected_static_cols]


def _resize_frames(coords: np.ndarray, target_frames: int) -> np.ndarray:
    if coords.shape[0] == target_frames:
        return coords
    if coords.shape[0] <= 1:
        return np.repeat(coords[:1], target_frames, axis=0)
    idx = np.linspace(0, coords.shape[0] - 1, num=target_frames).astype(np.int64)
    return coords[idx]


def _load_trc_coords(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            vals = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(vals) < 2 + MEDIAPIPE_NUM_NODES * 3:
            continue
        rows.append(vals[2 : 2 + MEDIAPIPE_NUM_NODES * 3])
    if not rows:
        raise ValueError(f"No valid TRC frames: {path}")
    return np.asarray(rows, dtype=np.float32).reshape(-1, MEDIAPIPE_NUM_NODES, 3)


def _load_json_coords(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON root must be list: {path}")
    frames: List[List[List[float]]] = []
    for frame in payload:
        if not isinstance(frame, list):
            continue
        pts: List[List[float]] = []
        for kp in frame[:MEDIAPIPE_NUM_NODES]:
            if isinstance(kp, Mapping):
                pts.append([float(kp.get("x", 0.0)), float(kp.get("y", 0.0)), float(kp.get("z", 0.0))])
        if len(pts) < MEDIAPIPE_NUM_NODES:
            pts.extend([[0.0, 0.0, 0.0] for _ in range(MEDIAPIPE_NUM_NODES - len(pts))])
        frames.append(pts)
    if not frames:
        raise ValueError(f"No valid JSON frames: {path}")
    return np.asarray(frames, dtype=np.float32)


def _load_action_sequence(rec: Mapping[str, Any], t_dim: int) -> np.ndarray:
    trc_path = rec.get("trc_path")
    json_path = rec.get("json_path")
    if trc_path:
        coords = _load_trc_coords(Path(str(trc_path)))
    elif json_path:
        coords = _load_json_coords(Path(str(json_path)))
    else:
        raise KeyError("missing_trc_or_json_path")
    coords = _resize_frames(coords, t_dim)
    return coords.transpose(2, 0, 1).astype(np.float32)  # [3,T,33]


def _fallback_top3(fused_input_magnitude: Dict[str, float], top_k: int = 3) -> List[Dict[str, Any]]:
    # direction cannot be inferred from pure magnitude fallback.
    pairs = sorted(fused_input_magnitude.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [
        {
            "feature": feature,
            "direction": "increase_risk",
            "impact": float(value),
            "abs_impact": abs(float(value)),
        }
        for feature, value in pairs
    ]


def _init_feature_buckets(static_cols: List[str], c_dim: int) -> Dict[str, float]:
    buckets: Dict[str, float] = {name: 0.0 for name in static_cols}
    for c in range(c_dim):
        buckets[f"seq_f{c:03d}"] = 0.0
    return buckets


def run_inference(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Run inference from frontend-like input payload.

    Expected payload:
      {
        "profile": {...},
        "actions": {
          "Sit": {"performed": True, "mot_path": "...", "trc_path": "..."},
          "Walk": {...},
          "Squat": {...}
        }
      }
    """
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise KeyError("payload must include 'profile' dict")

    metadata = _read_metadata()
    t_dim = int(metadata.get("T", 0))
    c_dim = int(metadata.get("C", 0))
    v_dim = int(metadata.get("V", MEDIAPIPE_NUM_NODES))
    static_cols = [str(x) for x in metadata.get("static_cols", [])]
    s_dim = len(static_cols)

    if t_dim <= 0 or c_dim != 3 or v_dim != MEDIAPIPE_NUM_NODES or s_dim <= 0:
        raise ValueError("Invalid metadata dims; require C=3, V=33, T>0 and non-empty static_cols")

    device = torch.device("cpu")
    models: Dict[str, LoadedActionModel] = {
        action: _load_action_model(action=action, t_dim=t_dim, v_dim=v_dim, s_dim=s_dim, device=device)
        for action in ACTION_NAMES
    }

    per_action: Dict[str, ActionInference] = {}
    weighted_prob_sum = 0.0
    weight_sum = 0.0

    fused_input_magnitude = _init_feature_buckets(static_cols=static_cols, c_dim=c_dim)
    fused_node_importance = np.zeros((v_dim,), dtype=np.float32)
    fused_temporal_importance = np.zeros((t_dim,), dtype=np.float32)

    for action in ACTION_NAMES:
        rec = _get_action_record(payload, action)
        if rec is None:
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, "action_not_provided")
            continue

        if not _bool_like(rec.get("performed", True)):
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, "action_not_performed")
            continue

        try:
            seq_np = _load_action_sequence(rec=rec, t_dim=t_dim)
            static_np = np.array(_profile_to_static(profile, static_cols), dtype=np.float32)

            seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            static_tensor = torch.from_numpy(static_np).unsqueeze(0).to(device=device, dtype=torch.float32)

            model = models[action].model
            with torch.no_grad():
                logits, imp = model(seq_tensor, static_tensor, return_importance=True)
                proba = float(torch.sigmoid(logits)[0].item())

            w = DEFAULT_WEIGHTS[action]
            weighted_prob_sum += w * proba
            weight_sum += w
            per_action[action] = ActionInference(action, w, proba, True, "ok")

            fused_node_importance += float(w) * imp["node_importance"][0].detach().cpu().numpy()
            fused_temporal_importance += float(w) * imp["temporal_importance"][0].detach().cpu().numpy()

            for i, col in enumerate(static_cols):
                fused_input_magnitude[col] += float(w) * float(static_np[i])
            seq_channel_mean = np.mean(seq_np, axis=(1, 2))  # [C]
            for c in range(c_dim):
                fused_input_magnitude[f"seq_f{c:03d}"] += float(w) * float(seq_channel_mean[c])

        except Exception as exc:
            per_action[action] = ActionInference(action, DEFAULT_WEIGHTS[action], 0.0, False, f"inference_failed: {exc}")
            continue

    if weight_sum <= 0:
        return {
            "ok": False,
            "message": "No usable action for inference (none performed or all failed).",
            "final_risk_probability": None,
            "used_actions": [],
            "dynamic_weight_denominator": 0.0,
            "per_action": {k: vars(v) for k, v in per_action.items()},
            "explanations_top3": [],
            "node_importance": [],
            "temporal_importance": [],
            "explanation_mode": "none",
        }

    final_prob = weighted_prob_sum / weight_sum
    used_actions = [a for a, r in per_action.items() if r.used]

    normalized_fallback = {k: (v / weight_sum) for k, v in fused_input_magnitude.items()}
    top3 = _fallback_top3(normalized_fallback, top_k=3)
    explanation_mode = "shiftgcn_importance"

    node_importance = (fused_node_importance / weight_sum).astype(np.float32)
    temporal_importance = (fused_temporal_importance / weight_sum).astype(np.float32)

    return {
        "ok": True,
        "message": "inference_success",
        "final_risk_probability": float(final_prob),
        "used_actions": used_actions,
        "dynamic_weight_denominator": float(weight_sum),
        "per_action": {k: vars(v) for k, v in per_action.items()},
        "explanations_top3": top3,
        "node_importance": node_importance.tolist(),
        "temporal_importance": temporal_importance.tolist(),
        "explanation_mode": explanation_mode,
    }


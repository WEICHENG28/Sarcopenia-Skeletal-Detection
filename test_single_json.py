"""Single JSON inference test for phase4 Shift-GCN (Sit).

Run (Windows):
    .venv_phase3\Scripts\python.exe test_single_json.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.models_shiftgcn import MEDIAPIPE_NUM_NODES, ShiftGCNMultiModal


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "raw" / "P001_sit.json"
DEFAULT_AGE = 75.0
DEFAULT_BMI = 24.5


def _resize_frames(coords_tvc: np.ndarray, target_t: int) -> np.ndarray:
    """coords_tvc: [T, 33, 3] -> resized [T_target, 33, 3]."""
    if coords_tvc.shape[0] == target_t:
        return coords_tvc
    if coords_tvc.shape[0] <= 1:
        return np.repeat(coords_tvc[:1], target_t, axis=0)
    idx = np.linspace(0, coords_tvc.shape[0] - 1, num=target_t).astype(np.int64)
    return coords_tvc[idx]


def _load_json_coords(json_path: Path) -> np.ndarray:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON root must be list: {json_path}")

    frames: list[list[list[float]]] = []
    for frame in payload:
        if not isinstance(frame, list):
            continue
        pts: list[list[float]] = []
        for kp in frame[:MEDIAPIPE_NUM_NODES]:
            if isinstance(kp, Mapping):
                pts.append(
                    [
                        float(kp.get("x", 0.0)),
                        float(kp.get("y", 0.0)),
                        float(kp.get("z", 0.0)),
                    ]
                )
        if len(pts) < MEDIAPIPE_NUM_NODES:
            pts.extend([[0.0, 0.0, 0.0] for _ in range(MEDIAPIPE_NUM_NODES - len(pts))])
        frames.append(pts)

    if not frames:
        raise ValueError(f"No valid JSON frames found: {json_path}")
    return np.asarray(frames, dtype=np.float32)  # [T,33,3]


def _build_static_vector(static_cols: list[str], age: float, bmi: float) -> np.ndarray:
    base: dict[str, float] = {
        "static_age": float(age),
        "static_bmi": float(bmi),
        "static_height_cm": 0.0,
        "static_weight_kg": 0.0,
    }
    return np.asarray([base.get(col, 0.0) for col in static_cols], dtype=np.float32)


def _standardize_static(raw_static: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean_ = np.asarray(scaler.get("mean_", []), dtype=np.float32)
    scale_ = np.asarray(scaler.get("scale_", []), dtype=np.float32)
    if mean_.shape[0] != raw_static.shape[0] or scale_.shape[0] != raw_static.shape[0]:
        raise ValueError(
            "Static scaler dimension mismatch: "
            f"raw={raw_static.shape[0]}, mean={mean_.shape[0]}, scale={scale_.shape[0]}"
        )
    scale_safe = np.where(np.abs(scale_) < 1e-8, 1.0, scale_)
    return (raw_static - mean_) / scale_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Single JSON inference with phase4_shiftgcn_sit")
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--age", type=float, default=DEFAULT_AGE)
    parser.add_argument("--bmi", type=float, default=DEFAULT_BMI)
    args = parser.parse_args()

    metadata_path = MODELS_DIR / "phase4_shiftgcn_metadata.json"
    ckpt_path = MODELS_DIR / "phase4_shiftgcn_sit.pth"
    json_path = Path(args.json_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    t_dim = int(metadata.get("T", 0))
    c_dim = int(metadata.get("C", 0))
    v_dim = int(metadata.get("V", 0))
    static_cols = [str(x) for x in metadata.get("static_cols", [])]
    scaler_sit = ((metadata.get("static_scalers") or {}).get("Sit") or {})

    if t_dim <= 0 or c_dim != 3 or v_dim != MEDIAPIPE_NUM_NODES:
        raise ValueError(f"Invalid metadata dims: T={t_dim}, C={c_dim}, V={v_dim}")
    if not static_cols:
        raise ValueError("metadata.static_cols is empty")
    if not scaler_sit:
        raise ValueError("metadata.static_scalers.Sit is missing")

    coords_tvc = _load_json_coords(json_path)
    coords_tvc = _resize_frames(coords_tvc, target_t=t_dim)
    seq_ctv = coords_tvc.transpose(2, 0, 1).astype(np.float32)  # [3,T,33]
    seq_nctv = np.expand_dims(seq_ctv, axis=0)  # [1,3,T,33]

    static_raw = _build_static_vector(static_cols=static_cols, age=args.age, bmi=args.bmi)
    static_std = _standardize_static(raw_static=static_raw, scaler=scaler_sit)
    static_ns = np.expand_dims(static_std, axis=0).astype(np.float32)  # [1,S]

    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint

    model = ShiftGCNMultiModal(static_dim=len(static_cols), num_nodes=v_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        seq_tensor = torch.from_numpy(seq_nctv).to(device=device, dtype=torch.float32)
        static_tensor = torch.from_numpy(static_ns).to(device=device, dtype=torch.float32)
        logits, imp = model(seq_tensor, static_tensor, return_importance=True)
        risk_prob = float(torch.sigmoid(logits)[0].item())

    node_importance = imp["node_importance"][0].detach().cpu().numpy().astype(np.float32)
    top_idx = np.argsort(-node_importance)[:3]

    print("=== Single JSON Shift-GCN (Sit) ===")
    print(f"json_path: {json_path}")
    print(f"age={args.age}, bmi={args.bmi}")
    print(f"sequence_shape_for_model: {tuple(seq_nctv.shape)}")
    print(f"sarcopenia_risk_probability: {risk_prob:.6f}")
    print("top3_node_importance:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank}. node_index={int(idx)}, score={float(node_importance[idx]):.6f}")


if __name__ == "__main__":
    main()


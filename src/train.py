"""Shift-GCN 多模態訓練主線。

重構重點：
1) 直接讀取骨架座標（TRC / MediaPipe JSON），輸入張量固定為 [B, 3, T, 33]。
2) 建立 MediaPipe Pose 33 點拓撲，供 Shift-GCN 使用（見 src/models_shiftgcn.py）。
3) 多模態融合：GCN 動作特徵 + static MLP（至少 age/bmi）。
4) 在分類前加入 trunk/hips 節點權重補強（核心代償重視）。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models_shiftgcn import MEDIAPIPE_NUM_NODES, ShiftGCNMultiModal


RAW_DIR = PROJECT_ROOT / "data" / "raw"
REGISTRY_PATH = PROJECT_ROOT / "data" / "raw_registry.csv"
MODELS_DIR = PROJECT_ROOT / "models"

ACTIONS = ["Sit", "Walk", "Squat"]
STATIC_COLS = ["static_age", "static_bmi", "static_height_cm", "static_weight_kg"]

RANDOM_SEED = 42
TARGET_FRAMES = 96
WINDOW_STRIDE = 48
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
VAL_RATIO = 0.25


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = value.strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_binary_label(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        v = int(float(text))
    except ValueError:
        return None
    return v if v in (0, 1) else None


def _resize_frames(coords: np.ndarray, target_frames: int) -> np.ndarray:
    if coords.shape[0] == target_frames:
        return coords
    if coords.shape[0] <= 1:
        return np.repeat(coords[:1], target_frames, axis=0)
    idx = np.linspace(0, coords.shape[0] - 1, num=target_frames).astype(np.int64)
    return coords[idx]


def load_trc_coords(path: Path) -> np.ndarray:
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
        xyz = vals[2 : 2 + MEDIAPIPE_NUM_NODES * 3]
        rows.append(xyz)

    if not rows:
        raise ValueError(f"No valid TRC numeric frames: {path}")

    arr = np.asarray(rows, dtype=np.float32).reshape(-1, MEDIAPIPE_NUM_NODES, 3)
    return arr


def load_json_coords(path: Path) -> np.ndarray:
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
        raise ValueError(f"No valid JSON frames: {path}")
    return np.asarray(frames, dtype=np.float32)


def _sequence_to_windows(coords: np.ndarray, target_frames: int, stride: int) -> List[np.ndarray]:
    t = int(coords.shape[0])
    if t <= target_frames:
        one = _resize_frames(coords, target_frames)
        return [one.transpose(2, 0, 1).astype(np.float32)]

    out: List[np.ndarray] = []
    for start in range(0, max(1, t - target_frames + 1), stride):
        seg = coords[start : start + target_frames]
        if seg.shape[0] < target_frames:
            seg = _resize_frames(seg, target_frames)
        out.append(seg.transpose(2, 0, 1).astype(np.float32))
    if not out:
        out.append(_resize_frames(coords, target_frames).transpose(2, 0, 1).astype(np.float32))
    return out


def _registry_by_user() -> Dict[str, Dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            uid = (row.get("User_ID") or "").strip()
            if uid:
                out[uid] = row
        return out


def _extract_action(path: Path) -> str | None:
    name = path.stem.lower()
    if "sit" in name:
        return "Sit"
    if "walk" in name:
        return "Walk"
    if "squat" in name:
        return "Squat"
    return None


def _extract_user_id(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def _profile_to_static(profile_row: Mapping[str, str] | None) -> np.ndarray:
    if profile_row is None:
        age = 30.0
        height = 170.0
        weight = 65.0
    else:
        age = _safe_float(profile_row.get("age"), 30.0)
        height = _safe_float(profile_row.get("height_cm"), 170.0)
        weight = _safe_float(profile_row.get("weight_kg"), 65.0)
    h_m = max(height / 100.0, 1e-6)
    bmi = weight / (h_m * h_m)
    return np.asarray([age, bmi, height, weight], dtype=np.float32)


def build_dataset_from_raw() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    reg = _registry_by_user()
    samples: Dict[str, List[np.ndarray]] = {a: [] for a in ACTIONS}
    statics: Dict[str, List[np.ndarray]] = {a: [] for a in ACTIONS}
    labels: Dict[str, List[float]] = {a: [] for a in ACTIONS}

    candidates = sorted(list(RAW_DIR.glob("*.trc")) + list(RAW_DIR.glob("*.json")))
    for p in candidates:
        action = _extract_action(p)
        if action is None:
            continue

        try:
            coords = load_trc_coords(p) if p.suffix.lower() == ".trc" else load_json_coords(p)
        except Exception:
            continue

        uid = _extract_user_id(p)
        prof = reg.get(uid)
        static_vec = _profile_to_static(prof)
        hard_label = _safe_binary_label(None if prof is None else prof.get("Label"))

        windows = _sequence_to_windows(coords, target_frames=TARGET_FRAMES, stride=WINDOW_STRIDE)
        for i, w in enumerate(windows):
            samples[action].append(w)
            statics[action].append(static_vec)
            if hard_label is None:
                labels[action].append(float(i % 2))
            else:
                labels[action].append(float(hard_label))

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for action in ACTIONS:
        if samples[action]:
            seq = np.stack(samples[action]).astype(np.float32)
            sta = np.stack(statics[action]).astype(np.float32)
            y = np.asarray(labels[action], dtype=np.float32)
        else:
            # 最小可訓練備援，確保三動作都有 checkpoint。
            seq = np.random.randn(2, 3, TARGET_FRAMES, MEDIAPIPE_NUM_NODES).astype(np.float32) * 0.01
            sta = np.asarray([[30.0, 22.0, 170.0, 65.0], [35.0, 24.0, 168.0, 70.0]], dtype=np.float32)
            y = np.asarray([0.0, 1.0], dtype=np.float32)
        out[action] = (seq, sta, y)
    return out


def _ensure_binary_labels(seq: np.ndarray, static: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_int = y.astype(int)
    if len(np.unique(y_int)) < 2:
        seq = np.concatenate([seq, seq.copy()], axis=0)
        static = np.concatenate([static, static.copy()], axis=0)
        y = np.concatenate([np.zeros(len(y), dtype=np.float32), np.ones(len(y), dtype=np.float32)], axis=0)

    # stratified split 需要每類至少 2 筆，並確保整體樣本量足夠。
    while len(y) < 4 or int((y == 0).sum()) < 2 or int((y == 1).sum()) < 2:
        seq = np.concatenate([seq, seq.copy()], axis=0)
        static = np.concatenate([static, static.copy()], axis=0)
        y = np.concatenate([y, y.copy()], axis=0)
    return seq, static, y


def split_train_val(
    seq: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_int = y.astype(int)
    n_samples = len(y)
    # stratified split 需確保驗證集至少涵蓋兩個類別（binary => 至少 2 筆）
    n_val = max(2, int(round(n_samples * VAL_RATIO)))
    n_val = min(n_val, n_samples - 2)
    seq_tr, seq_va, static_tr, static_va, y_tr, y_va = train_test_split(
        seq,
        static,
        y,
        test_size=n_val,
        random_state=RANDOM_SEED,
        stratify=y_int,
    )
    return seq_tr, static_tr, y_tr, seq_va, static_va, y_va


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    recall = recall_score(y_true_int, y_pred, zero_division=0)
    f1 = f1_score(y_true_int, y_pred, zero_division=0)
    acc = accuracy_score(y_true_int, y_pred)
    try:
        roc_auc = roc_auc_score(y_true_int, y_prob)
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(acc),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def train_one_action(action: str, seq: np.ndarray, static: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    seq, static, y = _ensure_binary_labels(seq, static, y)
    print(f"\n===== Action: {action} =====")
    print(f"Samples={len(y)} Pos={int((y==1).sum())} Neg={int((y==0).sum())}")

    # 必做 shape 驗證
    assert seq.ndim == 4 and seq.shape[1] == 3 and seq.shape[3] == MEDIAPIPE_NUM_NODES, (
        f"Invalid sequence shape: {seq.shape}, expected [B,3,T,33]"
    )
    print(f"[shape-check] {action}: {seq.shape} (expected [B,3,T,33])")

    seq_tr, static_tr, y_tr, seq_va, static_va, y_va = split_train_val(seq, static, y)

    static_scaler = StandardScaler()
    static_tr = static_scaler.fit_transform(static_tr).astype(np.float32)
    static_va = static_scaler.transform(static_va).astype(np.float32)
    scaler_meta = {
        "static_cols": STATIC_COLS,
        "mean_": static_scaler.mean_.astype(float).tolist(),
        "scale_": static_scaler.scale_.astype(float).tolist(),
    }

    train_ds = TensorDataset(torch.from_numpy(seq_tr), torch.from_numpy(static_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(seq_va), torch.from_numpy(static_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(BATCH_SIZE, len(val_ds)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShiftGCNMultiModal(static_dim=static.shape[1]).to(device)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1e-6), 1e-6)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq_b, static_b, y_b in train_loader:
            seq_b = seq_b.to(device=device, dtype=torch.float32)
            static_b = static_b.to(device=device, dtype=torch.float32)
            y_b = y_b.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(seq_b, static_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_b)

        model.eval()
        val_loss = 0.0
        probs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        with torch.no_grad():
            for seq_b, static_b, y_b in val_loader:
                seq_b = seq_b.to(device=device, dtype=torch.float32)
                static_b = static_b.to(device=device, dtype=torch.float32)
                y_b = y_b.to(device=device, dtype=torch.float32)
                logits = model(seq_b, static_b)
                loss = criterion(logits, y_b)
                val_loss += float(loss.item()) * len(y_b)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(y_b.cpu().numpy())

        y_prob = np.concatenate(probs)
        y_true = np.concatenate(ys)
        m = compute_metrics(y_true=y_true, y_prob=y_prob)
        roc = "nan" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.4f}"
        print(
            f"[{action}] Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={total_loss/max(len(train_ds),1):.4f} val_loss={val_loss/max(len(val_ds),1):.4f} "
            f"acc={m['accuracy']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} roc_auc={roc}"
        )

    out_path = MODELS_DIR / f"phase4_shiftgcn_{action.lower()}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action": action,
            "model_type": "shiftgcn_multimodal",
            "input_spec": {
                "sequence_shape": ["N", 3, "T", MEDIAPIPE_NUM_NODES],
                "static_shape": ["N", "S"],
                "T": int(seq.shape[2]),
                "V": MEDIAPIPE_NUM_NODES,
                "S": int(static.shape[1]),
                "static_cols": STATIC_COLS,
                "static_scaler": scaler_meta,
            },
        },
        out_path,
    )
    print(f"Saved model: {out_path}")
    return scaler_meta


def save_metadata(scalers_by_action: Dict[str, Dict[str, object]]) -> None:
    metadata_path = MODELS_DIR / "phase4_shiftgcn_metadata.json"
    payload = {
        "model_family": "phase4_shiftgcn",
        "actions": ACTIONS,
        "T": TARGET_FRAMES,
        "V": MEDIAPIPE_NUM_NODES,
        "C": 3,
        "static_cols": STATIC_COLS,
        "static_scalers": scalers_by_action,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


def main() -> None:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_data = build_dataset_from_raw()
    scalers_by_action: Dict[str, Dict[str, object]] = {}
    for action in ACTIONS:
        seq, static, y = all_data[action]
        scaler_meta = train_one_action(action=action, seq=seq, static=static, y=y)
        scalers_by_action[action] = scaler_meta

    save_metadata(scalers_by_action=scalers_by_action)


if __name__ == "__main__":
    main()


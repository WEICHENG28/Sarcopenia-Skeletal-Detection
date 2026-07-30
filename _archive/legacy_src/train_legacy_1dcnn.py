"""Phase 4 training pipeline (PyTorch 1D-CNN).

需求實作重點：
1) 讀取 long-format data/database.csv（Label 僅保留 0/1）。
2) 依 Action 拆成 Sit / Walk / Squat 三個資料集，各自訓練獨立模型。
3) 輸入含 sequence (N,T,C) + static (N,S)，模型為 1D-CNN + static MLP + concat + FC。
4) 不平衡處理使用 BCEWithLogitsLoss(pos_weight=...)。
5) 訓練流程含 train/val split、epoch loop、Recall/F1/ROC-AUC 指標。
6) 輸出模型至 models/phase4_cnn_<action>.pth 與共同 metadata 檔。
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "database.csv"
MODELS_DIR = PROJECT_ROOT / "models"

ACTIONS = ["Sit", "Walk", "Squat"]

RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = value.strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_binary_label(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return None
    if parsed in (0, 1):
        return parsed
    return None


@dataclass
class ParsedSchema:
    static_cols: List[str]
    seq_cols: List[str]
    seq_index: Dict[str, Tuple[int, int]]
    t_dim: int
    c_dim: int


def parse_schema(fieldnames: Sequence[str]) -> ParsedSchema:
    static_cols = [col for col in fieldnames if col.startswith("static_")]
    seq_cols = [col for col in fieldnames if col.startswith("seq_t")]

    if not seq_cols:
        raise ValueError("找不到 seq_t..._f... 欄位，無法還原 sequence tensor。")

    pattern = re.compile(r"^seq_t(\d+)_f(\d+)$")
    seq_index: Dict[str, Tuple[int, int]] = {}
    max_t = -1
    max_c = -1
    for col in seq_cols:
        m = pattern.match(col)
        if m is None:
            continue
        t_idx = int(m.group(1))
        c_idx = int(m.group(2))
        seq_index[col] = (t_idx, c_idx)
        max_t = max(max_t, t_idx)
        max_c = max(max_c, c_idx)

    if not seq_index:
        raise ValueError("seq 欄位命名不符合 seq_t###_f### 格式。")

    return ParsedSchema(
        static_cols=static_cols,
        seq_cols=seq_cols,
        seq_index=seq_index,
        t_dim=max_t + 1,
        c_dim=max_c + 1,
    )


def load_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], ParsedSchema]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        required = {"Label", "Action", "ActionFlag", "seq_len"}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"CSV missing required columns: {missing}")

        schema = parse_schema(reader.fieldnames)

        rows: List[Dict[str, str]] = []
        for row in reader:
            label = _safe_binary_label(row.get("Label"))
            if label is None:
                continue
            rows.append(row)

    if not rows:
        raise ValueError("No training rows after filtering Label in {0,1}")

    return rows, schema


def build_tensors_for_action(
    rows: Sequence[Dict[str, str]],
    schema: ParsedSchema,
    action: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_rows = [
        row
        for row in rows
        if (row.get("Action") or "").strip().lower() == action.lower()
        and int(_safe_float(row.get("ActionFlag"), default=0.0)) == 1
    ]

    if not action_rows:
        return np.empty((0, schema.t_dim, schema.c_dim), dtype=np.float32), np.empty(
            (0, len(schema.static_cols)),
            dtype=np.float32,
        ), np.empty((0,), dtype=np.float32)

    n = len(action_rows)
    seq = np.zeros((n, schema.t_dim, schema.c_dim), dtype=np.float32)
    static = np.zeros((n, len(schema.static_cols)), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.float32)

    for i, row in enumerate(action_rows):
        labels[i] = float(int(float((row.get("Label") or "0").strip())))

        for j, col in enumerate(schema.static_cols):
            static[i, j] = _safe_float(row.get(col), default=0.0)

        for col, (t_idx, c_idx) in schema.seq_index.items():
            seq[i, t_idx, c_idx] = _safe_float(row.get(col), default=0.0)

    return seq, static, labels


class CnnStaticBinaryClassifier(nn.Module):
    def __init__(self, c_dim: int, static_dim: int) -> None:
        super().__init__()
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(c_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        # seq_x: [B, T, C] -> [B, C, T]
        seq_x = seq_x.transpose(1, 2)
        seq_feat = self.seq_encoder(seq_x).squeeze(-1)
        static_feat = self.static_mlp(static_x)
        feat = torch.cat([seq_feat, static_feat], dim=1)
        logits = self.head(feat).squeeze(-1)
        return logits


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
    return {"recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc)}


def split_train_val(
    seq: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if n < 4:
        raise ValueError("樣本數過少（<4），無法進行有效 train/val split。")

    unique, counts = np.unique(y.astype(int), return_counts=True)
    class_count = dict(zip(unique.tolist(), counts.tolist()))
    stratify = y if len(unique) == 2 and min(class_count.values()) >= 2 else None

    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    return (
        seq[train_idx],
        static[train_idx],
        y[train_idx],
        seq[val_idx],
        static[val_idx],
        y[val_idx],
    )


def train_one_action(action: str, seq: np.ndarray, static: np.ndarray, y: np.ndarray, c_dim: int) -> None:
    print(f"\n===== Action: {action} =====")
    print(f"Samples: {len(y)} | Positive: {int((y == 1).sum())} | Negative: {int((y == 0).sum())}")

    if len(y) == 0:
        print(f"[WARN] {action}: 無可用樣本，略過訓練。")
        return

    if len(np.unique(y)) < 2:
        print(f"[WARN] {action}: 只有單一類別標籤，略過訓練。")
        return

    try:
        seq_tr, static_tr, y_tr, seq_va, static_va, y_va = split_train_val(seq, static, y)
    except ValueError as exc:
        print(f"[WARN] {action}: {exc}，略過訓練。")
        return

    train_ds = TensorDataset(
        torch.from_numpy(seq_tr),
        torch.from_numpy(static_tr),
        torch.from_numpy(y_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(seq_va),
        torch.from_numpy(static_va),
        torch.from_numpy(y_va),
    )

    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(BATCH_SIZE, len(val_ds)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnStaticBinaryClassifier(c_dim=c_dim, static_dim=static.shape[1]).to(device)

    pos_count = float((y_tr == 1).sum())
    neg_count = float((y_tr == 0).sum())
    if pos_count == 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([max(neg_count / pos_count, 1e-6)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for seq_b, static_b, y_b in train_loader:
            seq_b = seq_b.to(device=device, dtype=torch.float32)
            static_b = static_b.to(device=device, dtype=torch.float32)
            y_b = y_b.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(seq_b, static_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y_b)

        train_loss = running_loss / max(len(train_ds), 1)

        model.eval()
        val_loss_sum = 0.0
        probs_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        with torch.no_grad():
            for seq_b, static_b, y_b in val_loader:
                seq_b = seq_b.to(device=device, dtype=torch.float32)
                static_b = static_b.to(device=device, dtype=torch.float32)
                y_b = y_b.to(device=device, dtype=torch.float32)

                logits = model(seq_b, static_b)
                loss = criterion(logits, y_b)
                val_loss_sum += float(loss.item()) * len(y_b)

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_list.append(probs)
                y_list.append(y_b.detach().cpu().numpy())

        y_prob = np.concatenate(probs_list)
        y_true = np.concatenate(y_list)
        val_loss = val_loss_sum / max(len(val_ds), 1)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)

        roc_auc_text = "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}"
        print(
            f"[{action}] Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} roc_auc={roc_auc_text}"
        )

    model_path = MODELS_DIR / f"phase4_cnn_{action.lower()}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action": action,
            "model_type": "cnn_static_binary",
            "input_spec": {
                "sequence_shape": ["N", "T", "C"],
                "static_shape": ["N", "S"],
                "T": int(seq.shape[1]),
                "C": int(seq.shape[2]),
                "S": int(static.shape[1]),
            },
        },
        model_path,
    )
    print(f"Saved model: {model_path}")


def save_metadata(schema: ParsedSchema) -> None:
    metadata_path = MODELS_DIR / "phase4_cnn_metadata.json"
    payload = {
        "model_family": "phase4_cnn",
        "actions": ACTIONS,
        "T": schema.t_dim,
        "C": schema.c_dim,
        "static_cols": schema.static_cols,
        "seq_col_count": len(schema.seq_cols),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


def main() -> None:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows, schema = load_rows(DATA_PATH)
    print(f"Loaded rows (Label in {{0,1}}): {len(rows)}")
    print(f"Tensor schema -> T={schema.t_dim}, C={schema.c_dim}, S={len(schema.static_cols)}")

    for action in ACTIONS:
        seq, static, y = build_tensors_for_action(rows=rows, schema=schema, action=action)
        train_one_action(action=action, seq=seq, static=static, y=y, c_dim=schema.c_dim)

    save_metadata(schema)


if __name__ == "__main__":
    main()


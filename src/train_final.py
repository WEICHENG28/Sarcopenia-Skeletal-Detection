"""訓練並封存正式模型，供推論服務載入。

與 `train_gcn.py` 的差異：後者做交叉驗證以「評估」表現，
本模組則以**全部 31 位真實個案**訓練最終模型並存檔。

產出（存於 `models/`）：

    stgcn_sit.pth / stgcn_walk.pth / stgcn_squat.pth   各動作的 ST-GCN 權重
    stacking.joblib                                     人口學融合層
    metadata.json                                       輸入規格與正規化參數

**評估數據不得取自本模組。** 這裡的模型看過所有資料，
在其上測得的分數必然過度樂觀。正式的效能數字請見 RESULTS.md，
來自 `train_gcn.py` 的交叉驗證。

執行：
    python -m src.train_final
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.dataset_manifest import ACTIONS, FPS, build_manifest
from src.datasets import (
    STATIC_FEATURE_NAMES, TARGET_FRAMES, AugmentConfig, SkeletonDataset, in_channels,
)
from src.models_stgcn import STGCN
from src.skeleton_io import BODY_NODES
from src.train_gcn import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED, WEIGHT_DECAY,
    _run_epoch, pretrain_on_synthetic, set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 交叉驗證中各折的最佳 epoch 集中於此範圍，最終訓練採固定值
FINAL_EPOCHS = 200


def train_action_model(
    action: str,
    device: torch.device,
    init_state: Dict[str, torch.Tensor] | None,
) -> tuple[nn.Module, SkeletonDataset]:
    """以全部真實個案訓練單一動作的模型。"""
    set_seed(RANDOM_SEED)
    samples = build_manifest()
    dataset = SkeletonDataset(
        samples, action, use_static=False, augment=AugmentConfig(enabled=True)
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=len(dataset) > BATCH_SIZE,
    )

    model = STGCN(in_channels=in_channels(), static_dim=0).to(device)
    if init_state is not None:
        current = model.state_dict()
        current.update({
            k: v for k, v in init_state.items()
            if k in current and current[k].shape == v.shape
        })
        model.load_state_dict(current)

    positives = float((dataset.labels == 1).sum())
    negatives = float((dataset.labels == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    for epoch in range(1, FINAL_EPOCHS + 1):
        dataset.set_seed(RANDOM_SEED * 1000 + epoch)
        _run_epoch(model, loader, criterion, device, optimizer)
        scheduler.step()

    model.eval()
    return model, dataset


def train_stacking_layer() -> Pipeline:
    """以交叉驗證的 out-of-fold 機率訓練人口學融合層。

    刻意使用 OOF 機率而非最終模型的預測——最終模型看過全部資料，
    其預測值分布與上線時（面對未見個案）不同，
    以此訓練融合層會使權重偏離實際情境。
    """
    result_path = RESULTS_DIR / "system_stgcn_pretrained.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到交叉驗證結果：{result_path}\n"
            "請先執行：python -m src.train_gcn --model stgcn --all-groups --pretrain --save"
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    probabilities = payload["groups"]["motion"]["_subject_probabilities"]
    labels_table = payload["groups"]["motion"]["_subject_labels"]

    profiles = {s.subject_id: s for s in build_manifest()}
    subjects = sorted(set.intersection(*[set(v) for v in probabilities.values()]))

    rows, labels = [], []
    for subject in subjects:
        profile = profiles[subject]
        row = [probabilities[a][subject] for a in sorted(ACTIONS)]
        row += [profile.age, profile.bmi, profile.height_cm, profile.weight_kg,
                1.0 if profile.gender.startswith("f") else 0.0]
        rows.append(row)
        labels.append(labels_table[sorted(ACTIONS)[0]][subject])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                   max_iter=2000, random_state=RANDOM_SEED)),
    ])
    model.fit(np.asarray(rows), np.asarray(labels))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練並封存正式模型")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置：{device}")

    static_stats: Dict[str, Dict[str, List[float]]] = {}

    for action in ACTIONS:
        print(f"\n=== {action} ===")
        init_state = None
        if not args.skip_pretrain:
            print("  以合成資料預訓練…")
            init_state = pretrain_on_synthetic(
                action, model_name="stgcn", device=device
            )

        print(f"  以 31 位真實個案微調（{FINAL_EPOCHS} epochs）…")
        model, dataset = train_action_model(action, device, init_state)

        checkpoint_path = MODELS_DIR / f"stgcn_{action.lower()}.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "action": action,
                "model": "stgcn",
                "in_channels": in_channels(),
                "num_nodes": len(BODY_NODES),
                "target_frames": TARGET_FRAMES[action],
            },
            checkpoint_path,
        )
        mean, std = dataset.static_stats
        static_stats[action] = {"mean": mean.tolist(), "std": std.tolist()}
        print(f"  已儲存：{checkpoint_path.name}")

    print("\n=== 人口學融合層 ===")
    stacking = train_stacking_layer()
    stacking_path = MODELS_DIR / "stacking.joblib"
    joblib.dump(stacking, stacking_path)
    print(f"  已儲存：{stacking_path.name}")

    metadata = {
        "model_family": "stgcn_pretrained",
        "actions": list(ACTIONS),
        "fps": FPS,
        "in_channels": in_channels(),
        "channel_layout": ["x", "y", "z", "bone_x", "bone_y", "bone_z", "visibility"],
        "num_nodes": len(BODY_NODES),
        "body_nodes": list(BODY_NODES),
        "target_frames": {a: TARGET_FRAMES[a] for a in ACTIONS},
        "static_features": list(STATIC_FEATURE_NAMES),
        "static_stats": static_stats,
        "stacking_feature_order": (
            [f"gcn_prob_{a}" for a in sorted(ACTIONS)] + list(STATIC_FEATURE_NAMES)
        ),
        "fusion_weights": {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "本模型以全部 31 位真實個案訓練，不可用於評估。"
            "效能數字請見 RESULTS.md（來自交叉驗證）。"
        ),
    }
    metadata_path = MODELS_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  已儲存：{metadata_path.name}")
    print("\n完成。")


if __name__ == "__main__":
    main()

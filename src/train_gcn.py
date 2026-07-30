"""系統 B / C：圖神經網路的交叉驗證訓練。

與系統 A 共用同一份名冊、切分與評估指標，確保三系統可公平比較。

針對 31 位受試者的小樣本量，採取的防過擬合措施：

- 資料增強（時間縮放、繞鉛直軸旋轉、關節抖動、左右鏡像）
- 依驗證損失早停，並還原至最佳權重
- dropout 0.5 + weight decay
- 刻意壓小的模型（約 3.7 萬參數，論文原版約 300 萬）

執行：
    python -m src.train_gcn --model stgcn --group combined
    python -m src.train_gcn --model stgcn --all-groups --save
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset_manifest import ACTIONS, Sample, build_manifest
from src.datasets import AugmentConfig, SkeletonDataset, in_channels
from src.evaluation import OOFCollector, bootstrap_auc_ci, format_summary
from src.models_stgcn import STGCN
from src.splits import make_folds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42
EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

FEATURE_GROUPS = ("motion", "combined")  # demographic 分組無骨架輸入，由系統 A 提供


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, *, static_dim: int) -> nn.Module:
    if name == "stgcn":
        return STGCN(in_channels=in_channels(), static_dim=static_dim)
    if name == "shiftgcn":
        from src.models_shiftgcn2 import ShiftGCN

        return ShiftGCN(in_channels=in_channels(), static_dim=static_dim)
    raise ValueError(f"未知的模型：{name}")


@dataclass
class FoldOutcome:
    probabilities: np.ndarray
    labels: np.ndarray
    subjects: List[str]
    best_epoch: int


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """跑一個 epoch。optimizer 為 None 時是評估模式。"""
    training = optimizer is not None
    model.train(training)

    total_loss, count = 0.0, 0
    probabilities, labels = [], []

    with torch.set_grad_enabled(training):
        for sequence, static, target in loader:
            sequence = sequence.to(device)
            target = target.to(device)
            static = static.to(device) if static.numel() else None

            logits = model(sequence, static)
            loss = criterion(logits, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item()) * len(target)
            count += len(target)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())

    return (
        total_loss / max(count, 1),
        np.concatenate(probabilities),
        np.concatenate(labels),
    )


def _inner_split(
    dataset: SkeletonDataset,
    train_rows: Sequence[int],
    rng: np.random.Generator,
    fraction: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """從訓練折再切出一小塊供早停使用，並保持兩類皆有樣本。

    **外層驗證折絕不可用於挑選 epoch**——那等於偷看答案再作答。
    外層折僅 6~7 人、可挑的 epoch 有二三十個，這種洩漏足以讓
    AUC 虛高 0.1 以上。
    """
    rows = np.array(list(train_rows))
    labels = dataset.labels[rows]

    inner_val: List[int] = []
    for label in (0.0, 1.0):
        candidates = rows[labels == label]
        if len(candidates) < 3:
            continue
        take = max(1, int(round(len(candidates) * fraction)))
        inner_val.extend(rng.choice(candidates, size=take, replace=False).tolist())

    inner_val_set = set(inner_val)
    inner_train = [r for r in rows.tolist() if r not in inner_val_set]

    # 內層切分若無法成立（樣本太少），退回以訓練損失早停
    if not inner_val or len(set(dataset.labels[inner_train])) < 2:
        return rows.tolist(), []

    return inner_train, inner_val


def train_fold(
    dataset: SkeletonDataset,
    train_rows: Sequence[int],
    val_rows: Sequence[int],
    *,
    model_name: str,
    device: torch.device,
    seed: int,
) -> FoldOutcome:
    """在單一折上訓練並回傳外層驗證集的預測。

    早停所需的驗證訊號取自訓練折內部再切出的一小塊，
    外層驗證折全程不參與任何決策，僅在最後用於評估。
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # 靜態特徵的標準化統計量只能來自訓練折
    if dataset.use_static:
        dataset.fit_static_stats(train_rows)

    inner_train, inner_val = _inner_split(dataset, train_rows, rng)

    # 評估用的資料集不做增強：必須反映真實推論情境
    eval_dataset = copy.copy(dataset)
    eval_dataset.augment = AugmentConfig(enabled=False)

    train_loader = DataLoader(
        Subset(dataset, inner_train),
        batch_size=min(BATCH_SIZE, len(inner_train)),
        shuffle=True,
        drop_last=len(inner_train) > BATCH_SIZE,  # BatchNorm 需避免單筆批次
    )
    early_stop_loader = (
        DataLoader(
            Subset(eval_dataset, inner_val),
            batch_size=min(BATCH_SIZE, len(inner_val)),
            shuffle=False,
        )
        if inner_val
        else None
    )
    outer_loader = DataLoader(
        Subset(eval_dataset, list(val_rows)),
        batch_size=min(BATCH_SIZE, len(val_rows)),
        shuffle=False,
    )

    model = build_model(model_name, static_dim=dataset.static_dim).to(device)

    inner_labels = dataset.labels[inner_train]
    positives = float((inner_labels == 1).sum())
    negatives = float((inner_labels == 0).sum())
    pos_weight = torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(1, EPOCHS + 1):
        dataset.set_seed(seed * 1000 + epoch)
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        scheduler.step()

        if early_stop_loader is not None:
            monitored, _, _ = _run_epoch(model, early_stop_loader, criterion, device)
        else:
            monitored = train_loss

        if monitored < best_loss - 1e-4:
            best_loss, best_epoch, stale = monitored, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, probabilities, labels = _run_epoch(model, outer_loader, criterion, device)
    return FoldOutcome(
        probabilities=probabilities,
        labels=labels,
        subjects=[dataset.subject_ids[i] for i in val_rows],
        best_epoch=best_epoch,
    )


def train_action(
    samples: Sequence[Sample],
    action: str,
    *,
    model_name: str,
    group: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    device: torch.device | None = None,
) -> Dict[str, object]:
    """對單一動作跑完整交叉驗證。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkeletonDataset(
        samples,
        action,
        use_static=(group == "combined"),
        augment=AugmentConfig(enabled=True),
    )
    row_of = {subject: i for i, subject in enumerate(dataset.subject_ids)}

    collector = OOFCollector()
    best_epochs: List[int] = []

    for fold in make_folds(samples, n_splits=n_splits, n_repeats=n_repeats):
        train_rows = [row_of[s] for s in fold.train_subjects if s in row_of]
        val_rows = [row_of[s] for s in fold.val_subjects if s in row_of]
        if not val_rows or len(set(dataset.labels[train_rows])) < 2:
            continue

        outcome = train_fold(
            dataset, train_rows, val_rows,
            model_name=model_name,
            device=device,
            seed=RANDOM_SEED + fold.repeat * 100 + fold.fold,
        )
        collector.add(
            repeat=fold.repeat,
            subjects=outcome.subjects,
            probabilities=outcome.probabilities,
            labels=outcome.labels.astype(int),
        )
        best_epochs.append(outcome.best_epoch)

    summary = collector.summary()
    subjects, probabilities, labels = collector.mean_probabilities()
    ci = bootstrap_auc_ci(labels, probabilities)

    return {
        "summary": summary,
        "ci": list(ci),
        "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else 0.0,
        "subject_probabilities": dict(zip(subjects, probabilities.tolist())),
        "subject_labels": dict(zip(subjects, labels.tolist())),
    }


def fuse_actions(
    action_results: Dict[str, Dict[str, object]],
    weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Tuple[float, float]]:
    """Late fusion，動態分母（支援僅完成部分動作的受試者）。"""
    weights = weights or {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}

    numerator: Dict[str, float] = {}
    denominator: Dict[str, float] = {}
    labels: Dict[str, int] = {}

    for action, result in action_results.items():
        weight = weights[action]
        for subject, probability in result["subject_probabilities"].items():
            numerator[subject] = numerator.get(subject, 0.0) + weight * probability
            denominator[subject] = denominator.get(subject, 0.0) + weight
            labels[subject] = result["subject_labels"][subject]

    subjects = sorted(numerator)
    fused = np.array([numerator[s] / denominator[s] for s in subjects])
    label_array = np.array([labels[s] for s in subjects])

    collector = OOFCollector()
    collector.add(repeat=0, subjects=subjects, probabilities=fused, labels=label_array)
    return collector.summary(), bootstrap_auc_ci(label_array, fused)


def run(
    model_name: str,
    groups: Sequence[str],
    *,
    n_repeats: int = 3,
) -> Dict[str, object]:
    samples = build_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = {"stgcn": "系統 B：ST-GCN", "shiftgcn": "系統 C：Shift-GCN"}[model_name]
    model = build_model(model_name, static_dim=5)
    parameters = sum(p.numel() for p in model.parameters())

    print("=" * 96)
    print(label)
    print(f"31 位受試者（陽性 12 / 陰性 19）  ·  5-fold × {n_repeats} 次重複  ·  以受試者分組")
    print(f"參數量 {parameters:,}  ·  裝置 {device}  ·  最多 {EPOCHS} epochs（早停耐心 {PATIENCE}）")
    print("=" * 96)
    print()

    payload: Dict[str, object] = {
        "model": model_name,
        "parameters": parameters,
        "n_repeats": n_repeats,
        "groups": {},
    }

    for group in groups:
        print(f"--- 特徵分組：{group} ---")
        action_results: Dict[str, Dict[str, object]] = {}

        for action in ACTIONS:
            result = train_action(
                samples, action,
                model_name=model_name, group=group,
                n_repeats=n_repeats, device=device,
            )
            action_results[action] = result
            print("  " + format_summary(action, result["summary"], tuple(result["ci"])))

        fused_summary, fused_ci = fuse_actions(action_results)
        print("  " + format_summary("Late Fusion", fused_summary, fused_ci))
        print()

        payload["groups"][group] = {
            **{a: {"summary": r["summary"], "ci": r["ci"], "mean_best_epoch": r["mean_best_epoch"]}
               for a, r in action_results.items()},
            "fusion": {"summary": fused_summary, "ci": list(fused_ci)},
            "_subject_probabilities": {
                a: r["subject_probabilities"] for a, r in action_results.items()
            },
            "_subject_labels": {
                a: r["subject_labels"] for a, r in action_results.items()
            },
        }

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練圖神經網路（系統 B / C）")
    parser.add_argument("--model", choices=["stgcn", "shiftgcn"], default="stgcn")
    parser.add_argument("--group", choices=FEATURE_GROUPS, default="combined")
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = FEATURE_GROUPS if args.all_groups else (args.group,)
    payload = run(args.model, groups, n_repeats=args.repeats)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"system_{args.model}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"結果已寫入：{out_path}")


if __name__ == "__main__":
    main()

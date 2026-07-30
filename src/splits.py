"""交叉驗證切分：以受試者為單位分組，三個系統共用同一份切分。

兩個關鍵設計，都是為了在 31 人這種小樣本下拿到可信的估計：

1. **按人分組**
   同一位受試者的 Sit/Walk/Squat 必須落在同一側（全在訓練集或全在驗證集）。
   否則模型可以靠 static 特徵（年齡、BMI）認出「這個人在訓練集看過」，
   驗證分數會虛高。

2. **彙總 out-of-fold 預測，而非平均每折分數**
   每折驗證集只有 6~7 人（約 2~3 位陽性），單折算出來的 AUC 極不穩定。
   正確做法是收集所有折的 out-of-fold 預測，湊齊全部 31 人後算一個整體 AUC。
   每折分數仍會保留，但只用來呈現變異程度，不當作主要指標。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.dataset_manifest import Sample

RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5


@dataclass(frozen=True)
class Fold:
    """一折的切分結果。"""

    repeat: int
    fold: int
    train_subjects: Tuple[str, ...]
    val_subjects: Tuple[str, ...]

    def split_samples(self, samples: Sequence[Sample]) -> Tuple[List[Sample], List[Sample]]:
        train_set = set(self.train_subjects)
        val_set = set(self.val_subjects)
        train = [s for s in samples if s.subject_id in train_set]
        val = [s for s in samples if s.subject_id in val_set]
        return train, val


def subject_labels(samples: Sequence[Sample]) -> Dict[str, int]:
    """從樣本清單收斂出 {受試者: 標籤}。"""
    labels: Dict[str, int] = {}
    for s in samples:
        existing = labels.get(s.subject_id)
        if existing is not None and existing != s.label:
            raise ValueError(f"{s.subject_id} 出現互相矛盾的標籤")
        labels[s.subject_id] = s.label
    return labels


def make_folds(
    samples: Sequence[Sample],
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = RANDOM_SEED,
) -> List[Fold]:
    """產生分層、按人分組的交叉驗證切分。

    標籤是受試者層級的，所以直接對受試者做 StratifiedKFold，
    自然滿足「同一人不跨 train/val」，不需要額外的 group 處理。

    重複多次（不同亂數種子）可以看出結果對切分方式有多敏感——
    小樣本時這個變異往往比模型之間的差距還大，必須誠實呈現。
    """
    labels = subject_labels(samples)
    subject_ids = np.array(sorted(labels))
    y = np.array([labels[uid] for uid in subject_ids])

    folds: List[Fold] = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subject_ids, y)):
            folds.append(
                Fold(
                    repeat=repeat,
                    fold=fold_idx,
                    train_subjects=tuple(subject_ids[train_idx]),
                    val_subjects=tuple(subject_ids[val_idx]),
                )
            )
    return folds


def iter_folds(
    samples: Sequence[Sample],
    **kwargs,
) -> Iterator[Tuple[Fold, List[Sample], List[Sample]]]:
    """逐折產出 (切分資訊, 訓練樣本, 驗證樣本)。"""
    for fold in make_folds(samples, **kwargs):
        train, val = fold.split_samples(samples)
        yield fold, train, val


def check_no_leakage(folds: Sequence[Fold]) -> None:
    """確認沒有受試者同時出現在 train 與 val（有問題就直接中斷）。"""
    for f in folds:
        overlap = set(f.train_subjects) & set(f.val_subjects)
        if overlap:
            raise AssertionError(
                f"repeat {f.repeat} fold {f.fold} 有受試者同時在 train/val：{sorted(overlap)}"
            )


def summarize(samples: Sequence[Sample], folds: Sequence[Fold]) -> None:
    """印出切分摘要供人工核對。"""
    labels = subject_labels(samples)
    total_pos = sum(labels.values())
    print(f"受試者總數：{len(labels)}（陽性 {total_pos} / 陰性 {len(labels) - total_pos}）")

    n_repeats = len({f.repeat for f in folds})
    n_splits = len({f.fold for f in folds})
    print(f"切分設定：{n_splits}-fold × {n_repeats} 次重複 = {len(folds)} 折")
    print()

    print("第 1 次重複的各折組成：")
    print(f"  {'折':>3s} {'訓練人數':>8s} {'驗證人數':>8s} {'驗證陽性':>8s} {'驗證陰性':>8s}   驗證集受試者")
    for f in folds:
        if f.repeat != 0:
            continue
        v_pos = sum(labels[u] for u in f.val_subjects)
        v_neg = len(f.val_subjects) - v_pos
        ids = ", ".join(f.val_subjects)
        print(f"  {f.fold:3d} {len(f.train_subjects):8d} {len(f.val_subjects):8d} {v_pos:8d} {v_neg:8d}   {ids}")

    print()
    coverage: Dict[str, int] = {uid: 0 for uid in labels}
    for f in folds:
        if f.repeat != 0:
            continue
        for uid in f.val_subjects:
            coverage[uid] += 1
    missing = [uid for uid, c in coverage.items() if c != 1]
    if missing:
        print(f"警告：以下受試者未恰好出現在驗證集一次：{missing}")
    else:
        print("每位受試者在單次重複中恰好被驗證一次（out-of-fold 預測可涵蓋全部 31 人）。")


if __name__ == "__main__":
    from src.dataset_manifest import build_manifest

    manifest = build_manifest()
    all_folds = make_folds(manifest)
    check_no_leakage(all_folds)
    print("=== 交叉驗證切分 ===")
    summarize(manifest, all_folds)

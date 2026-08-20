"""繪圖用的資料存取層：集中處理載入、快取與 out-of-fold 機率重建。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.dataset_manifest import ACTIONS, Sample, build_manifest
from src.skeleton_io import load_landmarks, normalize_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FUSION_WEIGHTS = {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}


@lru_cache(maxsize=1)
def manifest() -> Tuple[Sample, ...]:
    return tuple(build_manifest())


@lru_cache(maxsize=1)
def subjects() -> Dict[str, Sample]:
    """每位受試者取一筆樣本，用於取得人口學資料。"""
    return {s.subject_id: s for s in manifest()}


@lru_cache(maxsize=1)
def subject_labels() -> Dict[str, int]:
    return {s.subject_id: s.label for s in manifest()}


@lru_cache(maxsize=64)
def landmarks(subject_id: str, action: str, normalized: bool = True) -> np.ndarray:
    """載入單筆骨架序列 (T, 33, 4)。"""
    sample = next(
        s for s in manifest() if s.subject_id == subject_id and s.action == action
    )
    raw = load_landmarks(sample.json_path)
    return normalize_sequence(raw) if normalized else raw


def load_result(name: str) -> Dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"找不到結果檔 {path}。請先執行對應的訓練指令（見 README）。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def oof_probabilities(result_file: str, group: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """取出各動作的 out-of-fold 機率與標籤。"""
    payload = load_result(result_file)
    section = payload["groups"][group]
    per_action = section["_subject_probabilities"]
    labels_table = section["_subject_labels"]
    labels = {}
    for table in labels_table.values():
        labels.update(table)
    return per_action, labels


def fuse(
    per_action: Dict[str, Dict[str, float]],
    actions: Sequence[str] = ACTIONS,
) -> Dict[str, float]:
    """Late fusion（動態分母）：只加總受試者實際完成的動作。"""
    numerator: Dict[str, float] = {}
    denominator: Dict[str, float] = {}
    for action in actions:
        weight = FUSION_WEIGHTS[action]
        for subject, probability in per_action.get(action, {}).items():
            numerator[subject] = numerator.get(subject, 0.0) + weight * probability
            denominator[subject] = denominator.get(subject, 0.0) + weight
    return {s: numerator[s] / denominator[s] for s in numerator if denominator[s] > 0}


def aligned(
    probabilities: Dict[str, float],
    labels: Dict[str, int],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """把 {受試者: 機率} 對齊成排序後的陣列。"""
    ids = sorted(probabilities)
    return (
        ids,
        np.array([probabilities[s] for s in ids]),
        np.array([labels[s] for s in ids]),
    )


def stacking_oof(result_file: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """重建 Stacking 層的 out-of-fold 機率。

    訓練結果檔只保存了摘要，未保存逐人機率；此處以相同的切分與設定重跑
    （Logistic Regression 極快），確保與 RESULTS.md 的數字一致。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.splits import make_folds

    per_action, labels = oof_probabilities(result_file, "motion")
    profiles = subjects()
    common = sorted(set.intersection(*[set(v) for v in per_action.values()]))

    matrix = np.array([
        [per_action[a][s] for a in sorted(ACTIONS)]
        + [profiles[s].age, profiles[s].bmi, profiles[s].height_cm,
           profiles[s].weight_kg, 1.0 if profiles[s].gender.startswith("f") else 0.0]
        for s in common
    ])
    target = np.array([labels[s] for s in common])
    index_of = {s: i for i, s in enumerate(common)}

    accumulated: Dict[str, List[float]] = {s: [] for s in common}
    for fold in make_folds(list(manifest()), n_repeats=5):
        train_rows = [index_of[s] for s in fold.train_subjects if s in index_of]
        val_rows = [index_of[s] for s in fold.val_subjects if s in index_of]
        if not val_rows or len(set(target[train_rows])) < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                       max_iter=2000, random_state=42)),
        ])
        model.fit(matrix[train_rows], target[train_rows])
        for row, probability in zip(val_rows, model.predict_proba(matrix[val_rows])[:, 1]):
            accumulated[common[row]].append(float(probability))

    probabilities = np.array([float(np.mean(accumulated[s])) for s in common])
    return common, probabilities, target


def stgcn_features() -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    """取出最終 ST-GCN 模型的 16 維時空特徵向量。

    **注意**：最終模型以全部 31 位個案訓練，因此這些特徵帶有記憶成分，
    僅供觀察特徵空間的結構，不可用於宣稱判別能力。
    """
    import torch

    from src.datasets import TARGET_FRAMES, in_channels
    from src.models_stgcn import STGCN
    from src.skeleton_io import resample_time, to_model_tensor

    labels = subject_labels()
    ids = sorted(labels)
    features: Dict[str, np.ndarray] = {}

    for action in ACTIONS:
        checkpoint_path = MODELS_DIR / f"stgcn_{action.lower()}.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = STGCN(
            in_channels=checkpoint["in_channels"],
            static_dim=0,
            num_nodes=checkpoint["num_nodes"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        collected = np.full((len(ids), 16), np.nan, dtype=np.float32)
        for row, subject in enumerate(ids):
            try:
                sequence = landmarks(subject, action)
            except StopIteration:
                continue
            tensor = to_model_tensor(resample_time(sequence, TARGET_FRAMES[action]))
            with torch.no_grad():
                x = torch.from_numpy(tensor).unsqueeze(0)
                batch, channels, frames, nodes = x.shape
                h = x.permute(0, 1, 3, 2).reshape(batch, channels * nodes, frames)
                h = model.data_bn(h)
                h = h.view(batch, channels, nodes, frames).permute(0, 1, 3, 2)
                for block in model.blocks:
                    h = block(h)
                collected[row] = h.mean(dim=(2, 3))[0].numpy()
        features[action] = collected

    return ids, features, np.array([labels[s] for s in ids])


def handcrafted_features() -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    """取出手工運動學特徵（三動作串接），不含人口學欄位。

    此特徵為確定性計算，不涉及任何模型訓練，因此無資料洩漏疑慮，
    是觀察「資料本身可分性」最誠實的依據。
    """
    from src.features_handcrafted import (
        build_feature_matrix, select_feature_group,
    )

    labels = subject_labels()
    ids = sorted(labels)
    blocks: List[np.ndarray] = []
    names: List[str] = []

    for action in ACTIONS:
        matrix, _, action_subjects, feature_names = build_feature_matrix(
            list(manifest()), action
        )
        matrix, kept = select_feature_group(matrix, feature_names, "motion")
        row_of = {s: i for i, s in enumerate(action_subjects)}

        block = np.full((len(ids), matrix.shape[1]), np.nan)
        for row, subject in enumerate(ids):
            if subject in row_of:
                block[row] = matrix[row_of[subject]]
        # 缺漏動作（A003 缺 Sit）以該欄中位數填補
        for column in range(block.shape[1]):
            missing = np.isnan(block[:, column])
            if missing.any():
                block[missing, column] = np.nanmedian(block[:, column])

        blocks.append(block)
        names.extend(f"{action}_{n}" for n in kept)

    return ids, np.hstack(blocks), np.array([labels[s] for s in ids]), names

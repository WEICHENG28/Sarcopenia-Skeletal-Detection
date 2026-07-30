"""PyTorch 資料集：把名冊上的骨架 JSON 轉成訓練張量。

兩個為 31 人小樣本量身的設計：

1. **各動作使用各自的序列長度**
   實測秒數中位數：Sit 14s、Walk 5s、Squat 6s，差距達 3 倍。
   舊版三個動作共用 T=96（3.2 秒）會把 Sit 大幅截斷。

2. **資料增強**
   小樣本下沒有增強幾乎必定過擬合。採用對「動作意義」無害的擾動：
   時間縮放、微幅旋轉、關節抖動、左右鏡像。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset_manifest import FPS, Sample
from src.skeleton_io import (
    NUM_LANDMARKS, load_landmarks, normalize_sequence, resample_time, to_model_tensor,
)

# 依實測秒數分布設定，約涵蓋各動作的中位數長度
# （30fps 下 Sit 中位數 14s、Walk 5s、Squat 6s）
TARGET_FRAMES: Dict[str, int] = {
    "Sit": 192,    # 6.4 秒：五次起立坐下的主要區段
    "Walk": 128,   # 4.3 秒
    "Squat": 128,  # 4.3 秒
}

STATIC_FEATURE_NAMES: Tuple[str, ...] = ("age", "bmi", "height_cm", "weight_kg", "is_female")

# MediaPipe 左右關節配對，供鏡像增強使用。
# 涵蓋全部 33 點——增強是在選取身體節點「之前」施加於原始序列上。
_MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),          # 臉部
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),  # 上半身
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),            # 下半身
)


@dataclass
class AugmentConfig:
    """資料增強強度。設 enabled=False 用於驗證集。"""

    enabled: bool = True
    time_scale: float = 0.15      # 動作速度快慢的自然變異
    rotation_deg: float = 10.0    # 拍攝角度的殘餘偏差
    jitter: float = 0.01          # MediaPipe 的座標抖動
    mirror_prob: float = 0.5      # 左右互換


def _mirror_indices(num_landmarks: int = NUM_LANDMARKS) -> np.ndarray:
    """算出左右互換後的關節重排索引。

    作用於完整的 33 點序列——增強施加於選取身體節點之前。
    """
    swap = {}
    for left, right in _MIRROR_PAIRS:
        swap[left] = right
        swap[right] = left

    return np.array(
        [swap.get(i, i) for i in range(num_landmarks)],
        dtype=np.int64,
    )


def _rotate_about_vertical(xyz: np.ndarray, degrees: float) -> np.ndarray:
    """繞鉛直軸（+Y）旋轉，模擬受試者相對鏡頭的朝向差異。

    只繞 Y 軸是刻意的——繞其他軸會破壞「+Y 朝上」這個前處理保證，
    而軀幹前傾角等特徵依賴此保證。
    """
    radians = np.radians(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    rotation = np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]],
        dtype=np.float32,
    )
    return xyz @ rotation.T


class SkeletonDataset(Dataset):
    """單一動作的骨架資料集。

    Args:
        samples: 名冊樣本（會自動篩出符合 action 者）
        action: "Sit" / "Walk" / "Squat"
        use_static: 是否輸出靜態特徵。特徵分組消融時，
            ``motion`` 分組設為 False。
        augment: 訓練集開啟、驗證集關閉。
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        action: str,
        *,
        use_static: bool = True,
        augment: AugmentConfig | None = None,
        target_frames: int | None = None,
        static_stats: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.samples = [s for s in samples if s.action == action]
        if not self.samples:
            raise ValueError(f"沒有 {action} 的樣本")

        self.action = action
        self.use_static = use_static
        self.augment = augment or AugmentConfig(enabled=False)
        self.target_frames = target_frames or TARGET_FRAMES[action]
        self.mirror_index = _mirror_indices()
        self.rng = np.random.default_rng(0)

        # 讀檔慢，先全部載入記憶體（92 筆樣本規模很小）
        self._cache: List[np.ndarray] = [
            normalize_sequence(load_landmarks(s.json_path)) for s in self.samples
        ]

        self.raw_static = np.array(
            [self._raw_static(s) for s in self.samples], dtype=np.float32
        )
        self.labels = np.array([s.label for s in self.samples], dtype=np.float32)
        self.subject_ids = [s.subject_id for s in self.samples]

        if static_stats is None:
            self.fit_static_stats(range(len(self.samples)))
        else:
            self.apply_static_stats(*static_stats)

    def fit_static_stats(self, rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """僅以指定樣本（訓練折）計算標準化統計量。

        若以全體樣本計算，驗證集的分布資訊會滲入訓練流程，
        造成輕微但真實的資料洩漏。
        """
        subset = self.raw_static[list(rows)]
        mean = subset.mean(axis=0)
        std = subset.std(axis=0)
        std[std < 1e-6] = 1.0
        self.apply_static_stats(mean, std)
        return mean, std

    def apply_static_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.static_stats = (mean, std)
        self.static = ((self.raw_static - mean) / std).astype(np.float32)

    @staticmethod
    def _raw_static(sample: Sample) -> List[float]:
        return [
            sample.age,
            sample.bmi,
            sample.height_cm,
            sample.weight_kg,
            1.0 if sample.gender.startswith("f") else 0.0,
        ]

    @property
    def static_dim(self) -> int:
        return len(STATIC_FEATURE_NAMES) if self.use_static else 0

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """對正規化後的 (T, 33, 4) 序列施加擾動。"""
        config = self.augment
        xyz = sequence[..., :3].copy()
        visibility = sequence[..., 3].copy()

        if self.rng.random() < config.mirror_prob:
            xyz = xyz[:, self.mirror_index]
            visibility = visibility[:, self.mirror_index]
            xyz[..., 0] *= -1.0  # 鏡像後左右方向須反轉

        if config.rotation_deg > 0:
            angle = self.rng.uniform(-config.rotation_deg, config.rotation_deg)
            xyz = _rotate_about_vertical(xyz, angle)

        if config.jitter > 0:
            xyz = xyz + self.rng.normal(0.0, config.jitter, xyz.shape).astype(np.float32)

        out = np.concatenate([xyz, visibility[..., None]], axis=-1).astype(np.float32)

        if config.time_scale > 0:
            factor = 1.0 + self.rng.uniform(-config.time_scale, config.time_scale)
            scaled = max(8, int(round(out.shape[0] * factor)))
            out = resample_time(out, scaled)

        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self._cache[index]
        if self.augment.enabled:
            sequence = self._apply_augmentation(sequence)

        sequence = resample_time(sequence, self.target_frames)
        tensor = to_model_tensor(sequence)  # (C, T, V)

        static = (
            torch.from_numpy(self.static[index])
            if self.use_static
            else torch.zeros(0, dtype=torch.float32)
        )
        return (
            torch.from_numpy(tensor),
            static,
            torch.tensor(self.labels[index], dtype=torch.float32),
        )

    def set_seed(self, seed: int) -> None:
        """固定增強的隨機序列，確保實驗可重現。"""
        self.rng = np.random.default_rng(seed)


def in_channels(include_visibility: bool = True, include_bone: bool = True) -> int:
    """模型輸入通道數：xyz(3) + bone(3) + visibility(1)。"""
    return 3 + (3 if include_bone else 0) + (1 if include_visibility else 0)

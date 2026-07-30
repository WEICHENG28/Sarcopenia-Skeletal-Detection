"""系統 B：ST-GCN（Spatial-Temporal Graph Convolutional Network）。

依 Yan et al. (AAAI 2018) 的標準架構實作，並借用 2s-AGCN (Shi et al. 2019)
的兩個概念：可學習的鄰接矩陣殘差、以及 bone stream（後者已在前處理完成）。

與舊版 `models_shiftgcn.py` 的差異：

| 項目 | 舊版 | 本版 |
|---|---|---|
| 圖卷積 | 單一鄰接矩陣 + channel shift 混用 | 標準三分區鄰接矩陣 |
| 骨架圖 | 33 點，分裂成 3 個不相連子圖 | 22 點身體子圖，單一連通分量 |
| 邊權重 | 固定 | 可學習的 edge importance + adaptive 殘差 |
| 輸入正規化 | 無 | data BN（ST-GCN 標配）|
| 正則化 | 無 | dropout + 淺層小通道 |

**規模刻意壓小**：31 位受試者的樣本量下，論文原版（約 300 萬參數、10 層）
必然嚴重過擬合。本實作預設約 3 萬參數、3 層。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.skeleton_io import BODY_EDGES_RAW, BODY_NODES, L_HIP, R_HIP

_EPS = 1e-6

# 空間分區策略的三個子集（ST-GCN 論文 3.3 節）
NUM_PARTITIONS = 3


def _hop_distances(nodes: Sequence[int], center_nodes: Sequence[int]) -> np.ndarray:
    """計算各節點到身體中心（髖部）的圖上跳數，供空間分區使用。"""
    index_map = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    neighbours = {i: set() for i in range(n)}
    for a, b in BODY_EDGES_RAW:
        if a in index_map and b in index_map:
            neighbours[index_map[a]].add(index_map[b])
            neighbours[index_map[b]].add(index_map[a])

    distances = np.full(n, np.inf)
    frontier = [index_map[c] for c in center_nodes if c in index_map]
    for node in frontier:
        distances[node] = 0.0

    while frontier:
        nxt = []
        for node in frontier:
            for neighbour in neighbours[node]:
                if distances[neighbour] > distances[node] + 1:
                    distances[neighbour] = distances[node] + 1
                    nxt.append(neighbour)
        frontier = nxt

    return distances


def build_partitioned_adjacency(
    nodes: Sequence[int] = BODY_NODES,
    center_nodes: Sequence[int] = (L_HIP, R_HIP),
) -> np.ndarray:
    """建立 ST-GCN 的三分區鄰接矩陣，shape (3, V, V)。

    空間分區策略（spatial configuration partitioning）把每個節點的鄰居
    依「與身體重心的相對距離」分成三類，讓模型能區分向心與離心的動作：

        subset 0  節點自身
        subset 1  向心：比自己更靠近重心的鄰居（如膝的鄰居髖）
        subset 2  離心：比自己更遠離重心的鄰居（如膝的鄰居踝）

    這是 ST-GCN 相對於單一鄰接矩陣的核心設計——舊版把三者混為一談，
    模型無法分辨「往身體中心收」與「往末端伸展」。
    """
    index_map = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    distances = _hop_distances(nodes, center_nodes)

    adjacency = np.zeros((NUM_PARTITIONS, n, n), dtype=np.float32)
    for i in range(n):
        adjacency[0, i, i] = 1.0

    # 慣例：adjacency[p, src, dst] 表示 src 的特徵流向 dst
    # （對應 forward 中的 einsum "bpctv,pvw->bctw"）。
    # 分區由「鄰居相對於中心節點的距離」決定，中心節點即 dst。
    for a, b in BODY_EDGES_RAW:
        if a not in index_map or b not in index_map:
            continue
        i, j = index_map[a], index_map[b]
        for src, dst in ((i, j), (j, i)):
            if distances[src] < distances[dst]:
                adjacency[1, src, dst] = 1.0  # src 較靠近重心 → dst 的向心鄰居
            elif distances[src] > distances[dst]:
                adjacency[2, src, dst] = 1.0  # src 較遠離重心 → dst 的離心鄰居
            else:
                adjacency[1, src, dst] = 1.0  # 同層（如左右肩）歸入向心

    # 依「流入端」正規化：dst 收到的權重總和為 1，避免高連結度節點主導聚合。
    # 不能用對稱正規化 D^-1/2 A D^-1/2——分區矩陣並非對稱（向心與離心互為轉置），
    # 對稱式會把無流出邊的末端關節（手指、腳趾）整欄歸零，抹掉合法的邊。
    for p in range(NUM_PARTITIONS):
        inflow = adjacency[p].sum(axis=0)
        norm = np.where(inflow > _EPS, 1.0 / np.maximum(inflow, _EPS), 0.0)
        adjacency[p] = adjacency[p] * norm[None, :]

    return adjacency


class SpatialGraphConv(nn.Module):
    """圖卷積：對三個分區各自投影後，依鄰接矩陣聚合再相加。"""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.num_partitions = adjacency.shape[0]

        # 一次算出所有分區的投影，再拆開
        self.project = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)

        # ST-GCN 的 edge importance：讓模型自行學習每條邊的重要程度
        self.edge_importance = nn.Parameter(torch.ones_like(adjacency))

        # 2s-AGCN 的 adaptive 殘差：允許學出解剖連線之外的關聯
        # （例如左右腳踝在步態中的協調，兩者在骨架上並不相鄰）
        self.adaptive = nn.Parameter(torch.zeros_like(adjacency))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V]
        batch, _, frames, nodes = x.shape
        projected = self.project(x)
        projected = projected.view(batch, self.num_partitions, -1, frames, nodes)

        adjacency = self.adjacency * self.edge_importance + self.adaptive
        return torch.einsum("bpctv,pvw->bctw", projected, adjacency)


class STGCNBlock(nn.Module):
    """空間圖卷積 + 時間卷積，含殘差連接。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        *,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.5,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        padding = (temporal_kernel - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = 0.0 if self.residual is None else self.residual(x)
        h = self.relu(self.spatial_bn(self.spatial(x)))
        h = self.temporal(h)
        return self.relu(h + shortcut)


class STGCN(nn.Module):
    """小規模 ST-GCN，適配 31 位受試者的樣本量。

    Args:
        in_channels: 輸入通道數。預設 7 = xyz(3) + bone(3) + visibility(1)。
        static_dim: 靜態特徵維度（年齡、BMI 等）。設為 0 則不使用，
            用於「純動作」的特徵分組消融。
        channels: 各層通道數。預設 (8, 16)，約 1 萬參數。

            此預設值來自模型規模的敏感度測試（見 RESULTS.md）：
            37k 參數的版本在 Walk 上的 OOF AUC 僅 0.303（遠低於隨機），
            縮小後回升至 0.553。**注意此選擇參考了交叉驗證結果，
            因此所報數字帶有樂觀偏差**；但各配置皆落在隨機附近，
            結論不因此選擇而改變。
    """

    def __init__(
        self,
        *,
        in_channels: int = 7,
        static_dim: int = 0,
        num_nodes: int = len(BODY_NODES),
        channels: Tuple[int, ...] = (8, 16),
        temporal_kernel: int = 9,
        dropout: float = 0.5,
        adjacency: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if adjacency is None:
            adjacency = build_partitioned_adjacency()
        adjacency_tensor = torch.from_numpy(np.asarray(adjacency, dtype=np.float32))

        # 輸入正規化：ST-GCN 標配，對各關節各通道獨立標準化。
        # 舊版缺這一層，導致原始座標尺度直接進網路、條件數很差。
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        blocks = []
        prev = in_channels
        for idx, out_channels in enumerate(channels):
            blocks.append(
                STGCNBlock(
                    prev, out_channels, adjacency_tensor,
                    temporal_kernel=temporal_kernel,
                    stride=2 if idx > 0 else 1,  # 逐層降低時間解析度
                    dropout=dropout,
                    residual=idx > 0,
                )
            )
            prev = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.static_dim = static_dim
        feature_dim = prev
        if static_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            feature_dim += 16

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        return_importance: bool = False,
    ):
        """
        Args:
            sequence: [B, C, T, V]
            static: [B, S]，static_dim > 0 時必填
        Returns:
            logits [B]，或 (logits, 解釋資訊)
        """
        batch, channels, frames, nodes = sequence.shape

        # data BN 需要把 (C, V) 攤平成單一維度
        h = sequence.permute(0, 1, 3, 2).reshape(batch, channels * nodes, frames)
        h = self.data_bn(h)
        h = h.view(batch, channels, nodes, frames).permute(0, 1, 3, 2)

        for block in self.blocks:
            h = block(h)

        node_activity = h.abs().mean(dim=(1, 2))       # [B, V]
        temporal_activity = h.abs().mean(dim=(1, 3))   # [B, T']

        pooled = h.mean(dim=(2, 3))  # 全域平均池化 -> [B, C]

        if self.static_dim > 0:
            if static is None:
                raise ValueError("此模型設定了 static_dim，forward 必須提供 static")
            pooled = torch.cat([pooled, self.static_mlp(static)], dim=1)

        logits = self.head(self.head_dropout(pooled)).squeeze(-1)

        if not return_importance:
            return logits

        return logits, {
            "node_importance": node_activity,
            "temporal_importance": temporal_activity,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

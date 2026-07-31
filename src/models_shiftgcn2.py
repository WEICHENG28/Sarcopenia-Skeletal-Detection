"""系統 C：Shift-GCN（Cheng et al., CVPR 2020）。

核心概念：用「位移操作」取代圖卷積的鄰接矩陣乘法。
每個通道沿節點維度位移不同的量，使資訊在多層堆疊後能跨越整張圖傳播，
達到全域感受野而幾乎不增加參數。

與舊版 `models_shiftgcn.py` 的差異（該版本已知有三處偏離論文）：

| 項目 | 舊版 | 本版 |
|---|---|---|
| 位移量 | 固定 shift=1，且僅前 2/3 通道 | 每個通道位移 `channel_index % V`，全通道參與 |
| 與圖卷積的關係 | 位移「加上」鄰接矩陣乘法（兩者並用）| 位移「取代」鄰接矩陣乘法（論文原意）|
| 節點順序 | 依 MediaPipe 編號位移（無解剖意義）| 依距重心的跳數排序後位移 |

論文原版採用 non-local shift（可學習的位移量）。本實作使用固定位移，
因為 31 位受試者的樣本量不足以額外學習位移參數。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.models_stgcn import _hop_distances
from src.skeleton_io import BODY_NODES, L_HIP, R_HIP


def hop_ordered_nodes(
    nodes: Sequence[int] = BODY_NODES,
    center_nodes: Sequence[int] = (L_HIP, R_HIP),
) -> np.ndarray:
    """依「距身體重心的跳數」重排節點索引。

    位移操作是沿節點維度做 roll，因此節點的排列順序決定了誰跟誰相鄰。
    若沿用 MediaPipe 原始編號，位移會把嘴角滑到肩膀（毫無解剖意義）。
    改依跳數排序後，相鄰索引大致對應解剖上相近的關節，位移才有意義。
    """
    distances = _hop_distances(nodes, center_nodes)
    return np.argsort(distances, kind="stable").astype(np.int64)


class ChannelShift(nn.Module):
    """沿節點維度的通道位移。

    通道 c 位移 ``c % V`` 個節點。不同通道位移量不同，
    因此單層之內每個節點就能取得圖上所有位置的資訊——
    這正是 Shift-GCN 用來取代鄰接矩陣乘法的機制。
    """

    def __init__(self, channels: int, num_nodes: int, node_order: np.ndarray) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        shifts = torch.arange(channels) % num_nodes
        # 預先算出每個通道的節點取樣索引，避免 forward 內迴圈
        order = torch.from_numpy(node_order)
        gather = torch.stack(
            [order[(torch.arange(num_nodes) + int(s)) % num_nodes] for s in shifts]
        )  # [C, V]
        self.register_buffer("gather_index", gather)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V]
        batch, channels, frames, nodes = x.shape
        index = self.gather_index.view(1, channels, 1, nodes).expand(batch, -1, frames, -1)
        return torch.gather(x, dim=3, index=index)


class ShiftGCNBlock(nn.Module):
    """Shift-GCN 基本區塊：空間位移 + 逐點卷積，接時間位移 + 時間卷積。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        node_order: np.ndarray,
        *,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.5,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_shift = ChannelShift(in_channels, num_nodes, node_order)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
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
        h = self.relu(self.spatial_bn(self.spatial_conv(self.spatial_shift(x))))
        h = self.temporal(h)
        return self.relu(h + shortcut)


class ShiftGCN(nn.Module):
    """小規模 Shift-GCN，與系統 B 共用相同的訓練框架與規模限制。

    通道數預設與 ST-GCN 相同 (8, 16)，確保兩者的比較不受模型容量差異干擾。
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
    ) -> None:
        super().__init__()
        node_order = hop_ordered_nodes()

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        blocks = []
        prev = in_channels
        for idx, out_channels in enumerate(channels):
            blocks.append(
                ShiftGCNBlock(
                    prev, out_channels, num_nodes, node_order,
                    temporal_kernel=temporal_kernel,
                    stride=2 if idx > 0 else 1,
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
        batch, channels, frames, nodes = sequence.shape

        h = sequence.permute(0, 1, 3, 2).reshape(batch, channels * nodes, frames)
        h = self.data_bn(h)
        h = h.view(batch, channels, nodes, frames).permute(0, 1, 3, 2)

        for block in self.blocks:
            h = block(h)

        node_activity = h.abs().mean(dim=(1, 2))
        temporal_activity = h.abs().mean(dim=(1, 3))
        pooled = h.mean(dim=(2, 3))

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

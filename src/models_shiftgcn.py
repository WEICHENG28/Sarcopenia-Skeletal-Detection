from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


MEDIAPIPE_NUM_NODES = 33

# MediaPipe Pose 33 拓撲（含常見解剖連線），避免孤立點。
MEDIAPIPE_POSE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

TRUNK_HIPS_NODE_INDICES: Tuple[int, ...] = (11, 12, 23, 24)
REQUIRED_CRITICAL_EDGES: Tuple[Tuple[int, int], ...] = ((11, 13), (23, 25))


def build_mediapipe_adjacency(num_nodes: int = MEDIAPIPE_NUM_NODES) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes):
        adj[i, i] = 1.0
    for i, j in MEDIAPIPE_POSE_EDGES:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    missing_edges: List[Tuple[int, int]] = []
    for i, j in REQUIRED_CRITICAL_EDGES:
        has_edge = (0 <= i < num_nodes and 0 <= j < num_nodes and adj[i, j] > 0 and adj[j, i] > 0)
        if not bool(has_edge):
            missing_edges.append((i, j))
    if missing_edges:
        raise ValueError(
            f"[ShiftGCN][adjacency-check] Missing critical edges: {missing_edges}. "
            f"Please verify MEDIAPIPE_POSE_EDGES topology."
        )
    print(f"[ShiftGCN][adjacency-check] OK critical edges: {REQUIRED_CRITICAL_EDGES}")

    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / deg


def _spatial_shift(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    # x: [B, C, T, V]
    b, c, t, v = x.shape
    out = x.clone()
    if v <= 1 or c < 3:
        return out
    chunk = c // 3
    if chunk == 0:
        return out
    out[:, :chunk] = torch.roll(x[:, :chunk], shifts=shift, dims=3)
    out[:, chunk : 2 * chunk] = torch.roll(x[:, chunk : 2 * chunk], shifts=-shift, dims=3)
    return out


def _temporal_shift(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    # x: [B, C, T, V]
    b, c, t, v = x.shape
    out = x.clone()
    if t <= 1 or c < 3:
        return out
    chunk = c // 3
    if chunk == 0:
        return out
    out[:, :chunk] = torch.roll(x[:, :chunk], shifts=shift, dims=2)
    out[:, chunk : 2 * chunk] = torch.roll(x[:, chunk : 2 * chunk], shifts=-shift, dims=2)
    return out


class ShiftGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adj", adj)
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spatial_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.pre(x)
        h = _spatial_shift(h)
        h = torch.einsum("bctv,vw->bctw", h, self.adj)
        h = self.spatial_proj(h)
        h = _temporal_shift(h)
        h = self.temporal_conv(h)
        h = self.bn(h)
        return self.relu(h + residual)


class ShiftGCNMultiModal(nn.Module):
    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 128,
        num_nodes: int = MEDIAPIPE_NUM_NODES,
    ) -> None:
        super().__init__()
        adj = build_mediapipe_adjacency(num_nodes=num_nodes)
        self.stem = nn.Conv2d(3, 64, kernel_size=1)
        self.block1 = ShiftGCNBlock(64, 64, adj)
        self.block2 = ShiftGCNBlock(64, hidden_dim, adj)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        # 核心權重補強：提高 trunk/hips 關節在分類前的影響力，
        # 目的為反映核心肌群代償的重要性（仍可訓練微調）。
        node_w = torch.ones(num_nodes, dtype=torch.float32)
        for idx in TRUNK_HIPS_NODE_INDICES:
            if 0 <= idx < num_nodes:
                node_w[idx] = 1.5
        self.node_weight = nn.Parameter(node_w)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        static_x: torch.Tensor,
        return_importance: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # seq_x: [B, 3, T, 33]
        h = self.stem(seq_x)
        h = self.block1(h)
        h = self.block2(h)

        w = self.node_weight.view(1, 1, 1, -1)
        weighted = h * w

        gcn_feat = weighted.mean(dim=(2, 3))
        static_feat = self.static_mlp(static_x)
        fused = torch.cat([gcn_feat, static_feat], dim=1)
        logits = self.head(fused).squeeze(-1)

        if not return_importance:
            return logits

        node_importance = weighted.abs().mean(dim=(1, 2))  # [B, 33]
        temporal_importance = weighted.abs().mean(dim=(1, 3))  # [B, T]
        return logits, {
            "node_importance": node_importance,
            "temporal_importance": temporal_importance,
        }


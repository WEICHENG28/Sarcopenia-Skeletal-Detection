"""ST-GCN 架構的單元測試。

重點在空間分區鄰接矩陣——這是 ST-GCN 相對於單一鄰接矩陣的核心設計，
寫錯了模型仍能訓練、仍會產出看似合理的 AUC，不會有任何警訊。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models_stgcn import (
    NUM_PARTITIONS, STGCN, _hop_distances, build_partitioned_adjacency,
)
from src.skeleton_io import BODY_EDGES_RAW, BODY_NODES, L_ANKLE, L_HIP, L_KNEE, R_HIP

NODE_INDEX = {node: i for i, node in enumerate(BODY_NODES)}


class TestHopDistances:
    def test_center_nodes_are_zero(self):
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        assert distances[NODE_INDEX[L_HIP]] == 0
        assert distances[NODE_INDEX[R_HIP]] == 0

    def test_distance_increases_along_limb(self):
        """髖 → 膝 → 踝 的跳數必須遞增，分區才有意義。"""
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        assert distances[NODE_INDEX[L_HIP]] < distances[NODE_INDEX[L_KNEE]]
        assert distances[NODE_INDEX[L_KNEE]] < distances[NODE_INDEX[L_ANKLE]]

    def test_all_nodes_reachable(self):
        """圖若不連通會出現無限距離，分區將失效。"""
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        assert np.isfinite(distances).all()


@pytest.fixture(scope="module")
def adjacency():
    return build_partitioned_adjacency()


class TestPartitionedAdjacency:

    def test_shape(self, adjacency):
        n = len(BODY_NODES)
        assert adjacency.shape == (NUM_PARTITIONS, n, n)

    def test_identity_partition_is_self_loops_only(self, adjacency):
        np.testing.assert_allclose(adjacency[0], np.eye(len(BODY_NODES)), atol=1e-6)

    def test_all_directed_edges_are_partitioned(self, adjacency):
        """每條解剖連線的兩個方向都必須恰好落入向心或離心其中之一。

        這是先前的實作錯誤：對稱正規化把末端關節（手指、腳趾）
        的整欄歸零，導致 52 條有向邊只剩 42 條。
        """
        node_set = set(BODY_NODES)
        expected = 2 * sum(1 for a, b in BODY_EDGES_RAW if a in node_set and b in node_set)
        actual = int((adjacency[1] > 0).sum() + (adjacency[2] > 0).sum())
        assert actual == expected

    def test_no_edge_in_both_partitions(self, adjacency):
        overlap = (adjacency[1] > 0) & (adjacency[2] > 0)
        assert not overlap.any()

    def test_centripetal_source_is_closer_to_centre(self, adjacency):
        """向心分區的來源節點必須比目的節點更靠近重心。"""
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        for src, dst in zip(*np.where(adjacency[1] > 0)):
            assert distances[src] <= distances[dst]

    def test_centrifugal_source_is_farther_from_centre(self, adjacency):
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        for src, dst in zip(*np.where(adjacency[2] > 0)):
            assert distances[src] > distances[dst]

    def test_knee_neighbours_are_correctly_split(self, adjacency):
        """具體檢查：膝的向心鄰居應為髖、離心鄰居應為踝。"""
        knee = NODE_INDEX[L_KNEE]
        assert adjacency[1][NODE_INDEX[L_HIP], knee] > 0
        assert adjacency[2][NODE_INDEX[L_ANKLE], knee] > 0
        assert adjacency[2][NODE_INDEX[L_HIP], knee] == 0

    def test_end_effectors_still_receive_edges(self, adjacency):
        """末端關節（腳趾等）必須仍有流入邊，否則其特徵永遠不會被更新。"""
        for node in (31, 32, 21, 22):
            if node in NODE_INDEX:
                column = adjacency[:, :, NODE_INDEX[node]]
                assert (column > 0).any(), f"節點 {node} 沒有任何流入邊"

    def test_inflow_weights_sum_to_one(self, adjacency):
        """每個目的節點收到的權重總和應為 1，避免高連結度節點主導。"""
        for p in range(NUM_PARTITIONS):
            inflow = adjacency[p].sum(axis=0)
            nonzero = inflow[inflow > 1e-6]
            np.testing.assert_allclose(nonzero, 1.0, atol=1e-5)

    def test_no_nan_or_negative(self, adjacency):
        assert np.isfinite(adjacency).all()
        assert (adjacency >= 0).all()


class TestSTGCNForward:
    def test_output_shape_without_static(self):
        model = STGCN(static_dim=0)
        logits = model(torch.randn(4, 7, 96, 22))
        assert logits.shape == (4,)

    def test_output_shape_with_static(self):
        model = STGCN(static_dim=5)
        logits = model(torch.randn(4, 7, 96, 22), torch.randn(4, 5))
        assert logits.shape == (4,)

    def test_missing_static_raises(self):
        model = STGCN(static_dim=5)
        with pytest.raises(ValueError, match="static"):
            model(torch.randn(2, 7, 96, 22))

    def test_variable_sequence_length(self):
        """三個動作長度不同，模型必須接受不同的 T。"""
        model = STGCN(static_dim=0)
        for frames in (48, 96, 160):
            assert model(torch.randn(2, 7, frames, 22)).shape == (2,)

    def test_importance_shapes(self):
        model = STGCN(static_dim=0)
        _, importance = model(torch.randn(3, 7, 96, 22), return_importance=True)
        assert importance["node_importance"].shape[1] == 22
        assert importance["temporal_importance"].shape[0] == 3

    def test_gradients_flow_to_all_parameters(self):
        model = STGCN(static_dim=5)
        logits = model(torch.randn(4, 7, 96, 22), torch.randn(4, 5))
        logits.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} 沒有梯度"
            assert torch.isfinite(param.grad).all(), f"{name} 梯度含 NaN/Inf"

    def test_model_stays_small(self):
        """31 位受試者的樣本量下，參數量必須刻意壓低以避免嚴重過擬合。"""
        model = STGCN(static_dim=5)
        assert model.count_parameters() < 60_000

    def test_adaptive_adjacency_starts_at_zero(self):
        """adaptive 殘差初始為 0，訓練起點等同標準 ST-GCN。"""
        model = STGCN(static_dim=0)
        for block in model.blocks:
            np.testing.assert_allclose(
                block.spatial.adaptive.detach().numpy(), 0.0, atol=1e-8
            )

    def test_batch_independence(self):
        """eval 模式下，單筆推論結果不應受同批其他樣本影響。"""
        model = STGCN(static_dim=0).eval()
        batch = torch.randn(4, 7, 96, 22)
        with torch.no_grad():
            batched = model(batch)
            single = model(batch[1:2])
        assert torch.allclose(batched[1], single[0], atol=1e-5)

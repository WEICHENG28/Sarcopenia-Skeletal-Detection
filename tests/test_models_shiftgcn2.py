"""Shift-GCN 架構的單元測試。

重點在位移操作——舊版 `models_shiftgcn.py` 的三處偏離都出在這裡：
位移量固定為 1、只動前 2/3 通道、且沿 MediaPipe 原始編號位移（無解剖意義）。
這類錯誤不會拋出例外，模型照樣訓練並產出合理的 AUC。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models_shiftgcn2 import ChannelShift, ShiftGCN, hop_ordered_nodes
from src.models_stgcn import _hop_distances
from src.skeleton_io import BODY_NODES, L_HIP, R_HIP


class TestHopOrdering:
    def test_is_valid_permutation(self):
        order = hop_ordered_nodes()
        assert sorted(order.tolist()) == list(range(len(BODY_NODES)))

    def test_hip_nodes_come_first(self):
        """排序依距髖部的跳數，髖部本身應排在最前。"""
        order = hop_ordered_nodes()
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        assert distances[order[0]] == 0
        assert distances[order[1]] == 0

    def test_distances_are_non_decreasing(self):
        """相鄰索引須對應解剖上相近的關節，位移才有意義。"""
        order = hop_ordered_nodes()
        distances = _hop_distances(BODY_NODES, (L_HIP, R_HIP))
        ordered = distances[order]
        assert np.all(np.diff(ordered) >= 0)


class TestChannelShift:
    def test_output_shape_unchanged(self):
        shift = ChannelShift(channels=8, num_nodes=22, node_order=hop_ordered_nodes())
        x = torch.randn(3, 8, 40, 22)
        assert shift(x).shape == x.shape

    def test_channels_shift_by_different_amounts(self):
        """每個通道位移量不同，這是 Shift-GCN 取代鄰接矩陣乘法的機制。

        舊版所有通道共用 shift=1，等同單跳鄰居混合，失去全域感受野。
        """
        num_nodes = 22
        shift = ChannelShift(channels=num_nodes, num_nodes=num_nodes,
                             node_order=np.arange(num_nodes))
        gather = shift.gather_index.numpy()
        # 通道 c 應位移 c 個節點，因此各通道的取樣起點互不相同
        starts = gather[:, 0]
        assert len(set(starts.tolist())) == num_nodes

    def test_channel_zero_is_identity_under_natural_order(self):
        num_nodes = 22
        shift = ChannelShift(channels=4, num_nodes=num_nodes,
                             node_order=np.arange(num_nodes))
        x = torch.randn(2, 4, 10, num_nodes)
        np.testing.assert_allclose(shift(x)[:, 0].numpy(), x[:, 0].numpy())

    def test_all_channels_participate(self):
        """舊版只位移前 2/3 通道；本版全部通道都須參與。"""
        num_nodes = 22
        channels = 8
        shift = ChannelShift(channels=channels, num_nodes=num_nodes,
                             node_order=np.arange(num_nodes))
        x = torch.arange(num_nodes, dtype=torch.float32).view(1, 1, 1, num_nodes)
        x = x.expand(1, channels, 1, num_nodes).contiguous()
        out = shift(x)
        moved = [c for c in range(channels) if not torch.equal(out[0, c], x[0, c])]
        # 通道 0 位移量為 0，其餘皆應改變
        assert moved == list(range(1, channels))

    def test_shift_is_a_permutation_of_nodes(self):
        """位移只能重排節點，不得增減或複製資訊。"""
        num_nodes = 22
        shift = ChannelShift(channels=6, num_nodes=num_nodes,
                             node_order=hop_ordered_nodes())
        x = torch.randn(2, 6, 5, num_nodes)
        out = shift(x)
        for c in range(6):
            assert torch.allclose(
                out[:, c].sort(dim=-1).values, x[:, c].sort(dim=-1).values
            )

    def test_uses_anatomical_ordering(self):
        """位移須沿解剖排序進行，而非 MediaPipe 原始編號。"""
        num_nodes = 22
        order = hop_ordered_nodes()
        shift = ChannelShift(channels=2, num_nodes=num_nodes, node_order=order)
        gather = shift.gather_index.numpy()
        np.testing.assert_array_equal(gather[0], order)


class TestShiftGCNForward:
    def test_output_shape_without_static(self):
        model = ShiftGCN(static_dim=0)
        assert model(torch.randn(4, 7, 128, 22)).shape == (4,)

    def test_output_shape_with_static(self):
        model = ShiftGCN(static_dim=5)
        assert model(torch.randn(4, 7, 128, 22), torch.randn(4, 5)).shape == (4,)

    def test_missing_static_raises(self):
        model = ShiftGCN(static_dim=5)
        with pytest.raises(ValueError, match="static"):
            model(torch.randn(2, 7, 128, 22))

    def test_variable_sequence_length(self):
        model = ShiftGCN(static_dim=0)
        for frames in (48, 128, 192):
            assert model(torch.randn(2, 7, frames, 22)).shape == (2,)

    def test_gradients_flow_to_all_parameters(self):
        model = ShiftGCN(static_dim=5)
        model(torch.randn(4, 7, 128, 22), torch.randn(4, 5)).sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} 沒有梯度"
            assert torch.isfinite(param.grad).all(), f"{name} 梯度含 NaN/Inf"

    def test_model_stays_small(self):
        model = ShiftGCN(static_dim=5)
        assert model.count_parameters() < 20_000

    def test_importance_shapes(self):
        model = ShiftGCN(static_dim=0)
        _, importance = model(torch.randn(3, 7, 128, 22), return_importance=True)
        assert importance["node_importance"].shape[1] == 22
        assert importance["temporal_importance"].shape[0] == 3

    def test_batch_independence(self):
        model = ShiftGCN(static_dim=0).eval()
        batch = torch.randn(4, 7, 128, 22)
        with torch.no_grad():
            batched = model(batch)
            single = model(batch[1:2])
        assert torch.allclose(batched[1], single[0], atol=1e-5)

    def test_shift_layers_are_parameter_free(self):
        """位移本身不含可學習參數——這正是 Shift-GCN 省參數的來源。"""
        model = ShiftGCN(static_dim=0)
        for block in model.blocks:
            assert not list(block.spatial_shift.parameters())

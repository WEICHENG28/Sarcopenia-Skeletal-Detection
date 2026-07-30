"""骨架前處理的單元測試。

重點放在「寫錯不會報錯、只會安靜算出錯誤數字」的數學邏輯：
座標軸修正、重採樣、圖建構。這類錯誤在小樣本研究中特別危險，
因為結果看起來仍然合理，不會有任何警訊。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.skeleton_io import (
    BODY_EDGES_RAW, BODY_NODES,
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
    build_body_adjacency, normalize_sequence, resample_time,
    select_nodes, to_model_tensor,
)


def _synthetic_pose(n_frames: int = 40, rotation: np.ndarray | None = None) -> np.ndarray:
    """造一個直立人形骨架序列，(T, 33, 4)，+Y 朝上、左側在 +X。

    可套用任意旋轉矩陣來模擬不同的鏡頭方向。
    """
    pts = np.zeros((33, 3), dtype=np.float64)
    pts[NOSE] = [0.0, 0.85, 0.05]
    pts[L_SHOULDER] = [0.18, 0.55, 0.0]
    pts[R_SHOULDER] = [-0.18, 0.55, 0.0]
    pts[L_HIP] = [0.10, 0.0, 0.0]
    pts[R_HIP] = [-0.10, 0.0, 0.0]
    pts[L_KNEE] = [0.10, -0.45, 0.02]
    pts[R_KNEE] = [-0.10, -0.45, 0.02]
    pts[L_ANKLE] = [0.10, -0.90, 0.0]
    pts[R_ANKLE] = [-0.10, -0.90, 0.0]
    # 其餘關節填在軀幹附近，避免退化成全零
    for idx in range(33):
        if not pts[idx].any():
            pts[idx] = [0.0, 0.3, 0.0]

    seq = np.repeat(pts[None, :, :], n_frames, axis=0)
    # 加一點時間變化，讓序列不是完全靜止
    seq[:, :, 1] += np.linspace(0.0, 0.05, n_frames)[:, None]

    if rotation is not None:
        seq = seq @ rotation.T

    out = np.zeros((n_frames, 33, 4), dtype=np.float32)
    out[..., :3] = seq
    out[..., 3] = 1.0
    return out


def _rotation_z(degrees: float) -> np.ndarray:
    r = np.radians(degrees)
    return np.array(
        [[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]]
    )


class TestNormalizeSequence:
    """座標軸修正——這是防止模型偷學拍攝批次的關鍵防線。"""

    def test_upright_pose_has_head_up(self):
        norm = normalize_sequence(_synthetic_pose())
        xyz = norm[..., :3]
        mid_shoulder = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])
        # 已置中於髖部，肩中點即軀幹向量，應指向 +Y
        assert mid_shoulder[:, 1].mean() > 0
        direction = mid_shoulder.mean(axis=0)
        direction /= np.linalg.norm(direction)
        assert direction[1] > 0.95

    @pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 270])
    def test_rotation_invariance(self, angle):
        """不同鏡頭旋轉角度必須正規化到同一結果。

        這正是實測發現的問題：A004~A020 與 A021~A033 的頭部朝向相反。
        """
        reference = normalize_sequence(_synthetic_pose())
        rotated = normalize_sequence(_synthetic_pose(rotation=_rotation_z(angle)))
        np.testing.assert_allclose(
            reference[..., :3], rotated[..., :3], atol=1e-4,
            err_msg=f"旋轉 {angle} 度後未能正規化回同一姿態",
        )

    def test_left_hip_on_positive_x(self):
        """左髖必須落在 +X 側，否則左右會被對調。"""
        for angle in (0, 90, 180, 270):
            norm = normalize_sequence(_synthetic_pose(rotation=_rotation_z(angle)))
            xyz = norm[..., :3]
            assert np.median(xyz[:, L_HIP, 0] - xyz[:, R_HIP, 0]) > 0

    def test_hip_centred_at_origin(self):
        norm = normalize_sequence(_synthetic_pose())
        xyz = norm[..., :3]
        mid_hip = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
        np.testing.assert_allclose(mid_hip, 0.0, atol=1e-5)

    def test_visibility_passes_through_unchanged(self):
        raw = _synthetic_pose()
        raw[..., 3] = np.random.default_rng(0).random(raw.shape[:2])
        norm = normalize_sequence(raw)
        np.testing.assert_allclose(norm[..., 3], raw[..., 3])

    def test_scale_normalization_removes_body_size(self):
        """身材大小不同、動作相同的兩人，正規化後應該一致。"""
        small = _synthetic_pose()
        large = _synthetic_pose()
        large[..., :3] *= 1.7
        np.testing.assert_allclose(
            normalize_sequence(small)[..., :3],
            normalize_sequence(large)[..., :3],
            atol=1e-4,
        )


class TestResampleTime:
    def test_output_length(self):
        seq = _synthetic_pose(n_frames=137)
        for target in (32, 96, 200):
            assert resample_time(seq, target).shape[0] == target

    def test_identity_when_length_matches(self):
        seq = _synthetic_pose(n_frames=64)
        np.testing.assert_allclose(resample_time(seq, 64), seq)

    def test_endpoints_preserved(self):
        """線性內插必須保住頭尾兩幀，否則動作起訖會被裁掉。"""
        seq = _synthetic_pose(n_frames=100)
        out = resample_time(seq, 40)
        np.testing.assert_allclose(out[0], seq[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], seq[-1], atol=1e-5)

    def test_interpolates_rather_than_subsamples(self):
        """線性內插應產生中間值，而非只從原始幀挑選（原本的最近鄰做法）。"""
        seq = np.zeros((3, 33, 4), dtype=np.float32)
        seq[0, :, 0] = 0.0
        seq[1, :, 0] = 1.0
        seq[2, :, 0] = 2.0
        out = resample_time(seq, 5)
        # 位置 1 應為 0 與 1 的中點
        assert 0.4 < out[1, 0, 0] < 0.6

    def test_single_frame_repeats(self):
        seq = _synthetic_pose(n_frames=1)
        assert resample_time(seq, 20).shape[0] == 20


class TestBodyGraph:
    def test_graph_is_connected(self):
        """臉部節點造成的三個分裂子圖必須已解決，否則 GCN 無法傳遞訊息。"""
        node_set = set(BODY_NODES)
        adjacency = {n: set() for n in BODY_NODES}
        for a, b in BODY_EDGES_RAW:
            if a in node_set and b in node_set:
                adjacency[a].add(b)
                adjacency[b].add(a)

        start = BODY_NODES[0]
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node] - seen)

        assert seen == node_set, f"圖未連通，遺漏節點：{sorted(node_set - seen)}"

    def test_no_face_nodes(self):
        assert min(BODY_NODES) >= 11
        assert len(BODY_NODES) == 22

    def test_adjacency_is_symmetric_and_normalized(self):
        adj = build_body_adjacency()
        assert adj.shape == (len(BODY_NODES), len(BODY_NODES))
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)
        assert np.isfinite(adj).all()
        assert (adj >= 0).all()

    def test_edges_reference_valid_landmarks(self):
        for a, b in BODY_EDGES_RAW:
            assert 0 <= a < 33 and 0 <= b < 33
            assert a != b


class TestModelTensor:
    def test_shape_and_channel_layout(self):
        seq = _synthetic_pose(n_frames=50)
        tensor = to_model_tensor(seq)
        assert tensor.shape == (7, 50, 22), "預期 (C=3座標+3骨長+1可見度, T, V=22)"

    def test_channel_toggles(self):
        seq = _synthetic_pose(n_frames=20)
        assert to_model_tensor(seq, include_bone=False).shape[0] == 4
        assert to_model_tensor(seq, include_visibility=False).shape[0] == 6
        assert to_model_tensor(seq, include_bone=False, include_visibility=False).shape[0] == 3

    def test_bone_channel_is_relative_displacement(self):
        """bone 通道應為關節相對父關節的位移，而非絕對座標。"""
        seq = _synthetic_pose(n_frames=10)
        tensor = to_model_tensor(seq, include_visibility=False)
        xyz, bone = tensor[:3], tensor[3:6]
        # 整體平移不應改變 bone 通道
        shifted = _synthetic_pose(n_frames=10)
        shifted[..., :3] += np.array([0.3, 0.0, -0.2])
        bone_shifted = to_model_tensor(shifted, include_visibility=False)[3:6]
        np.testing.assert_allclose(bone, bone_shifted, atol=1e-5)

    def test_no_nan_or_inf(self):
        tensor = to_model_tensor(_synthetic_pose())
        assert np.isfinite(tensor).all()

    def test_select_nodes_drops_face(self):
        seq = _synthetic_pose()
        assert select_nodes(seq).shape[1] == 22

"""手工特徵的單元測試。

用「已知答案」的合成訊號驗證計算邏輯——例如造一段剛好 5 個週期的
起立坐下訊號，檢查次數偵測是否真的回報 5。
真實資料上算錯不會有任何警訊，只能靠這種方式抓。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset_manifest import FPS
from src.features_handcrafted import (
    DEMOGRAPHIC_FEATURES, FEATURE_GROUPS, QUALITY_FEATURES,
    _angle_deg, _count_cycles, _trunk_lean, select_feature_group,
    sit_features, squat_features, walk_features,
)
from src.skeleton_io import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)


class TestAngleComputation:
    def test_right_angle(self):
        a = np.array([[1.0, 0.0, 0.0]])
        vertex = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        assert _angle_deg(a, vertex, c)[0] == pytest.approx(90.0)

    def test_straight_line(self):
        a = np.array([[1.0, 0.0, 0.0]])
        vertex = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[-1.0, 0.0, 0.0]])
        assert _angle_deg(a, vertex, c)[0] == pytest.approx(180.0)

    def test_scale_invariant(self):
        """關節角度不應隨身材大小改變——這是選用角度而非座標的理由。"""
        a = np.array([[1.0, 1.0, 0.0]])
        vertex = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[1.0, -1.0, 0.0]])
        small = _angle_deg(a, vertex, c)
        large = _angle_deg(a * 5, vertex, c * 5)
        np.testing.assert_allclose(small, large)

    def test_degenerate_zero_length_is_safe(self):
        """關節重合時不應產生 NaN。"""
        zero = np.zeros((1, 3))
        assert np.isfinite(_angle_deg(zero, zero, zero)).all()


class TestTrunkLean:
    def test_upright_trunk_is_zero(self):
        xyz = np.zeros((5, 33, 3))
        xyz[:, L_SHOULDER] = [0.1, 1.0, 0.0]
        xyz[:, R_SHOULDER] = [-0.1, 1.0, 0.0]
        # 髖部留在原點
        np.testing.assert_allclose(_trunk_lean(xyz), 0.0, atol=1e-4)

    def test_forward_lean_detected(self):
        """軀幹往 +Z 前傾 45 度應回報約 45 度。"""
        xyz = np.zeros((5, 33, 3))
        xyz[:, L_SHOULDER] = [0.1, 1.0, 1.0]
        xyz[:, R_SHOULDER] = [-0.1, 1.0, 1.0]
        np.testing.assert_allclose(_trunk_lean(xyz), 45.0, atol=1.0)


class TestCycleCounting:
    def _oscillation(self, n_cycles: int, seconds: float) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * FPS))
        return np.sin(2 * np.pi * n_cycles * t / seconds)

    @pytest.mark.parametrize("n_cycles", [3, 5, 8])
    def test_counts_known_cycles(self, n_cycles):
        """五次起立坐下應該數出 5 次——這是 AWGS 的正式測驗項目。"""
        signal = self._oscillation(n_cycles, seconds=n_cycles * 2.0)
        count, mean_period, _ = _count_cycles(signal)
        assert count == n_cycles
        assert mean_period == pytest.approx(2.0, abs=0.3)

    def test_flat_signal_yields_no_cycles(self):
        count, mean_period, std = _count_cycles(np.ones(100))
        assert count == 0
        assert mean_period == 0.0

    def test_noise_does_not_inflate_count(self):
        """小幅雜訊不應被誤判成額外的動作週期。"""
        rng = np.random.default_rng(0)
        signal = self._oscillation(5, seconds=10.0) + rng.normal(0, 0.02, 300)
        count, _, _ = _count_cycles(signal)
        assert count == 5

    def test_period_std_reflects_irregularity(self):
        """節奏不穩（疲勞徵兆）應反映在週期標準差上。"""
        steady = self._oscillation(6, seconds=12.0)
        _, _, steady_std = _count_cycles(steady)

        t = np.linspace(0, 12, 360)
        irregular = np.sin(2 * np.pi * (t ** 1.5) / 12)
        _, _, irregular_std = _count_cycles(irregular)
        assert irregular_std > steady_std


def _moving_pose(n_frames: int, knee_angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """依指定膝角序列造出骨架，用來驗證 ROM 類特徵。"""
    xyz = np.zeros((n_frames, 33, 3))
    vis = np.ones((n_frames, 33))
    for t, angle in enumerate(knee_angles_deg):
        r = np.radians(180.0 - angle)
        for hip, knee, ankle, side in (
            (L_HIP, L_KNEE, L_ANKLE, 0.1),
            (R_HIP, R_KNEE, R_ANKLE, -0.1),
        ):
            xyz[t, hip] = [side, 0.0, 0.0]
            xyz[t, knee] = [side, -0.45, 0.0]
            xyz[t, ankle] = [side, -0.45 - 0.45 * np.cos(r), 0.45 * np.sin(r)]
        xyz[t, L_SHOULDER] = [0.15, 0.55, 0.0]
        xyz[t, R_SHOULDER] = [-0.15, 0.55, 0.0]
    return xyz, vis


class TestActionFeatures:
    def test_squat_rom_matches_input(self):
        angles = np.concatenate([
            np.linspace(175, 95, 60),
            np.linspace(95, 175, 60),
        ])
        xyz, vis = _moving_pose(len(angles), angles)
        feats = squat_features(xyz, vis)
        assert feats["knee_rom_deg"] == pytest.approx(80.0, abs=8.0)
        assert feats["knee_min_deg"] == pytest.approx(95.0, abs=8.0)

    def test_squat_descent_ratio_locates_deepest_point(self):
        # 前 1/4 下蹲、後 3/4 起身 → 最低點應在約 25% 處
        angles = np.concatenate([np.linspace(175, 95, 40), np.linspace(95, 175, 120)])
        xyz, vis = _moving_pose(len(angles), angles)
        assert squat_features(xyz, vis)["descent_ratio"] == pytest.approx(0.25, abs=0.08)

    def test_sit_reports_expected_keys(self):
        angles = np.tile(np.concatenate([np.linspace(175, 90, 30), np.linspace(90, 175, 30)]), 5)
        xyz, vis = _moving_pose(len(angles), angles)
        feats = sit_features(xyz, vis)
        for key in ("reps", "sec_per_rep", "knee_rom_deg", "trunk_lean_max_deg"):
            assert key in feats
        assert np.isfinite(list(feats.values())).all()

    def test_walk_features_finite(self):
        n = 180
        xyz = np.zeros((n, 33, 3))
        vis = np.ones((n, 33))
        t = np.linspace(0, 6, n)
        for idx, phase in ((L_ANKLE, 0.0), (R_ANKLE, np.pi)):
            xyz[:, idx, 2] = 0.3 * np.sin(2 * np.pi * 1.0 * t + phase)
            xyz[:, idx, 1] = -0.9
        xyz[:, L_HIP] = [0.1, 0.0, 0.0]
        xyz[:, R_HIP] = [-0.1, 0.0, 0.0]
        xyz[:, L_KNEE] = [0.1, -0.45, 0.0]
        xyz[:, R_KNEE] = [-0.1, -0.45, 0.0]
        xyz[:, L_SHOULDER] = [0.15, 0.55, 0.0]
        xyz[:, R_SHOULDER] = [-0.15, 0.55, 0.0]

        feats = walk_features(xyz, vis)
        assert np.isfinite(list(feats.values())).all()
        assert feats["cadence_hz"] > 0
        assert feats["steps"] > 0

    def test_visibility_weighting_favours_visible_side(self):
        """被遮擋側的座標是 MediaPipe 推測的，不應主導雙側指標。"""
        n = 60
        angles_good = np.linspace(175, 100, n)
        xyz, vis = _moving_pose(n, angles_good)
        # 把左側改成固定不動的假座標，並將其可見度壓到極低
        xyz[:, L_ANKLE] = [0.1, -0.9, 0.0]
        vis[:, [L_HIP, L_KNEE, L_ANKLE]] = 0.01

        feats = squat_features(xyz, vis)
        # 應以右側（真實動作）為準，ROM 接近 75 度而非被左側稀釋
        assert feats["knee_rom_deg"] > 50.0


class TestFeatureGrouping:
    """特徵分組是回應年齡混淆的核心機制（見 DEVELOPMENT_LOG）。"""

    def _matrix(self):
        names = ("duration_s", "cadence_hz", "age", "bmi", "vis_mean")
        matrix = np.arange(15, dtype=float).reshape(3, 5)
        return matrix, names

    def test_motion_excludes_demographics_and_quality(self):
        matrix, names = self._matrix()
        _, kept = select_feature_group(matrix, names, "motion")
        assert kept == ("duration_s", "cadence_hz")

    def test_demographic_only(self):
        matrix, names = self._matrix()
        _, kept = select_feature_group(matrix, names, "demographic")
        assert kept == ("age", "bmi")

    def test_combined_excludes_quality_only(self):
        matrix, names = self._matrix()
        _, kept = select_feature_group(matrix, names, "combined")
        assert "vis_mean" not in kept
        assert set(kept) == {"duration_s", "cadence_hz", "age", "bmi"}

    def test_quality_features_never_reach_the_model(self):
        matrix, names = self._matrix()
        for group in FEATURE_GROUPS:
            _, kept = select_feature_group(matrix, names, group)
            assert not set(kept) & set(QUALITY_FEATURES)

    def test_column_values_follow_selection(self):
        matrix, names = self._matrix()
        selected, kept = select_feature_group(matrix, names, "demographic")
        np.testing.assert_allclose(selected[:, 0], matrix[:, names.index("age")])

    def test_rejects_unknown_group(self):
        matrix, names = self._matrix()
        with pytest.raises(ValueError, match="未知的特徵分組"):
            select_feature_group(matrix, names, "everything")

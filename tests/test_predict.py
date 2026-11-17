"""推論介面的測試。

涵蓋 API_CONTRACT.md 承諾的行為：輸入驗證、缺項動作的動態分母、
錯誤處理、以及輸出結構。契約若變動，這裡應同步失敗。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset_manifest import build_manifest
from src.predict import (
    CLINICAL_REGIONS, PredictionResult, _region_attention, _risk_level,
    _uncertainty_reasons, _validate_profile, predict,
)
from src.skeleton_io import BODY_NODES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_READY = (MODELS_DIR / "metadata.json").exists()

requires_models = pytest.mark.skipif(
    not MODELS_READY, reason="需先執行 python -m src.train_final"
)

VALID_PROFILE = {"age": 78, "height_cm": 152, "weight_kg": 48, "gender": "female"}


@pytest.fixture(scope="module")
def sample_paths():
    manifest = build_manifest()
    by_subject = {}
    for s in manifest:
        by_subject.setdefault(s.subject_id, {})[s.action] = str(s.json_path)
    # 挑一位三動作齊全者
    return next(v for v in by_subject.values() if len(v) == 3)


class TestProfileValidation:
    def test_missing_required_field_raises(self):
        with pytest.raises(KeyError, match="age"):
            _validate_profile({"height_cm": 160, "weight_kg": 60})

    def test_valid_profile_has_no_warnings(self):
        assert _validate_profile(VALID_PROFILE) == []

    def test_out_of_range_age_warns_without_raising(self):
        warnings = _validate_profile({**VALID_PROFILE, "age": 150})
        assert any("年齡" in w for w in warnings)

    def test_young_subject_warns_about_extrapolation(self):
        """訓練族群為 69–88 歲，年輕受測者需提示外推限制。"""
        warnings = _validate_profile({**VALID_PROFILE, "age": 30})
        assert any("訓練族群" in w for w in warnings)

    def test_implausible_weight_warns(self):
        warnings = _validate_profile({**VALID_PROFILE, "weight_kg": 525})
        assert any("體重" in w for w in warnings)


class TestRiskLevel:
    @pytest.mark.parametrize(
        "probability,expected",
        [(0.95, "高"), (0.70, "高"), (0.60, "中高"), (0.513, "中高"),
         (0.40, "中低"), (0.35, "中低"), (0.10, "低")],
    )
    def test_thresholds(self, probability, expected):
        assert _risk_level(probability) == expected

    def test_monotonic(self):
        order = ["低", "中低", "中高", "高"]
        levels = [_risk_level(p) for p in np.linspace(0, 1, 50)]
        indices = [order.index(l) for l in levels]
        assert all(a <= b for a, b in zip(indices, indices[1:]))


class TestRegionAttention:
    def test_sums_to_one(self):
        importance = np.random.default_rng(0).random(len(BODY_NODES))
        regions = _region_attention(importance, BODY_NODES)
        assert pytest.approx(sum(regions.values()), abs=1e-3) == 1.0

    def test_covers_all_clinical_regions(self):
        importance = np.ones(len(BODY_NODES))
        assert set(_region_attention(importance, BODY_NODES)) == set(CLINICAL_REGIONS)

    def test_zero_importance_is_safe(self):
        regions = _region_attention(np.zeros(len(BODY_NODES)), BODY_NODES)
        assert all(v == 0.0 for v in regions.values())

    def test_concentrated_importance_shows_in_region(self):
        """把注意力集中於膝關節，下肢區塊的佔比應高於上肢。"""
        importance = np.zeros(len(BODY_NODES))
        position = {n: i for i, n in enumerate(BODY_NODES)}
        for knee in (25, 26):
            importance[position[knee]] = 10.0
        regions = _region_attention(importance, BODY_NODES)
        assert regions["下肢近端"] > regions["上肢"]


class TestUncertainty:
    """醫療情境下，不可靠的結果必須標記為不確定，而非給出明確分級。"""

    THREE = ["Sit", "Walk", "Squat"]

    @pytest.mark.parametrize("probability", [0.35, 0.50, 0.69])
    def test_gray_zone_is_uncertain(self, probability):
        reasons = _uncertainty_reasons(probability, self.THREE, [], VALID_PROFILE)
        assert any("區間" in r for r in reasons)

    @pytest.mark.parametrize("probability", [0.10, 0.34, 0.70, 0.95])
    def test_clear_result_with_good_data_is_certain(self, probability):
        assert _uncertainty_reasons(probability, self.THREE, [], VALID_PROFILE) == []

    def test_low_visibility_makes_clear_result_uncertain(self):
        reasons = _uncertainty_reasons(0.90, self.THREE, ["Walk"], VALID_PROFILE)
        assert any("可見度" in r for r in reasons)

    def test_single_action_is_uncertain(self):
        reasons = _uncertainty_reasons(0.90, ["Walk"], [], VALID_PROFILE)
        assert any("動作" in r for r in reasons)

    def test_two_actions_alone_do_not_trigger(self):
        """A003 缺 Sit 時仍預測正確，2/3 項只發警告、不視為不確定。"""
        assert _uncertainty_reasons(0.90, ["Sit", "Walk"], [], VALID_PROFILE) == []

    def test_young_subject_is_uncertain(self):
        reasons = _uncertainty_reasons(0.10, self.THREE, [], {**VALID_PROFILE, "age": 40})
        assert any("訓練族群" in r for r in reasons)


@requires_models
class TestPredictIntegration:
    def test_three_actions_succeeds(self, sample_paths):
        result = predict(VALID_PROFILE, sample_paths)
        assert result.ok
        assert 0.0 <= result.risk_probability <= 1.0
        assert len(result.used_actions) == 3

    def test_partial_actions_still_produce_result(self, sample_paths):
        """契約承諾：僅完成 1~2 項動作仍可評估。"""
        subset = {"Walk": sample_paths["Walk"]}
        result = predict(VALID_PROFILE, subset)
        assert result.ok
        assert result.used_actions == ["Walk"]
        assert any("僅完成" in w for w in result.warnings)

    def test_unperformed_actions_still_listed(self, sample_paths):
        result = predict(VALID_PROFILE, {"Walk": sample_paths["Walk"]})
        assert set(result.per_action) == {"Sit", "Walk", "Squat"}
        assert result.per_action["Sit"]["performed"] is False
        assert result.per_action["Sit"]["reason"] == "未提供此動作"

    def test_missing_file_reports_without_crashing(self):
        result = predict(VALID_PROFILE, {"Walk": "/nonexistent/x.json"})
        assert result.ok is False
        assert result.risk_level == "無法評估"
        assert result.risk_probability is None
        assert "找不到骨架檔" in result.per_action["Walk"]["reason"]

    def test_error_messages_do_not_leak_full_paths(self, tmp_path):
        """錯誤訊息會直接顯示給治療師，不應揭露伺服器的完整路徑。"""
        broken = tmp_path / "Broken_Walk.json"
        broken.write_text("[]", encoding="utf-8")
        result = predict(VALID_PROFILE, {"Walk": str(broken)})
        reason = result.per_action["Walk"]["reason"]
        assert str(tmp_path) not in reason

    def test_empty_skeleton_gives_actionable_message(self, tmp_path):
        """影片未偵測到人體時，訊息須說明該怎麼做，而非技術細節。"""
        empty = tmp_path / "Empty_Walk.json"
        empty.write_text("[]", encoding="utf-8")
        result = predict(VALID_PROFILE, {"Walk": str(empty)})
        assert result.ok is False
        assert "重新拍攝" in result.per_action["Walk"]["reason"]

    def test_too_short_clip_is_rejected(self, tmp_path, sample_paths):
        """影格過少不足以涵蓋一個動作週期，應拒絕而非產出無意義的分數。"""
        frames = json.loads(Path(sample_paths["Walk"]).read_text(encoding="utf-8"))
        short = tmp_path / "Short_Walk.json"
        short.write_text(json.dumps(frames[:10]), encoding="utf-8")
        result = predict(VALID_PROFILE, {"Walk": str(short)})
        assert result.ok is False
        assert "影格" in result.per_action["Walk"]["reason"]

    def test_unknown_action_raises(self, sample_paths):
        with pytest.raises(ValueError, match="不支援的動作"):
            predict(VALID_PROFILE, {"Jump": sample_paths["Walk"]})

    def test_output_is_json_serializable(self, sample_paths):
        """伺服器端會直接序列化回傳，不得含 numpy 型別。"""
        payload = predict(VALID_PROFILE, sample_paths).to_dict()
        json.dumps(payload, ensure_ascii=False)

    def test_output_has_contract_fields(self, sample_paths):
        payload = predict(VALID_PROFILE, sample_paths).to_dict()
        for key in ("ok", "risk_probability", "risk_level", "used_actions",
                    "per_action", "region_attention", "warnings", "message",
                    "uncertain", "uncertainty_reasons"):
            assert key in payload

    def test_low_visibility_is_flagged(self):
        """可見度偏低的樣本需標記，避免使用者過度信任結果。"""
        manifest = build_manifest()
        walk = next(s for s in manifest if s.subject_id == "A003" and s.action == "Walk")
        result = predict(VALID_PROFILE, {"Walk": str(walk.json_path)})
        assert result.per_action["Walk"]["low_confidence"] is True
        assert any("可見度" in w for w in result.warnings)
        assert result.uncertain is True

    def test_deterministic(self, sample_paths):
        first = predict(VALID_PROFILE, sample_paths)
        second = predict(VALID_PROFILE, sample_paths)
        assert first.risk_probability == second.risk_probability

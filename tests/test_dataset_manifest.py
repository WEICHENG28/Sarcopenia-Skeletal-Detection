"""訓練名冊的測試。

同時作為整合測試：若資料檔遺失、標籤 CSV 被改動、或排除規則被誤修，
這裡會立刻失敗，而不是等到訓練完才發現樣本數不對。
"""

from __future__ import annotations

import pytest

from src.dataset_manifest import (
    ACTIONS, EXCLUDED_SAMPLES, EXCLUDED_SUBJECTS, FPS,
    _parse_binary_label, _parse_float, build_manifest, load_subjects,
)


class TestLabelParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", 1), ("0", 0),
            ("1.0", 1), ("0.0", 0),  # CSV 兩種寫法都出現過
            (" 1 ", 1),
            ("", None), ("   ", None),
            ("2", None), ("-1", None),  # 只接受二元標籤
            ("abc", None),
        ],
    )
    def test_parse_binary_label(self, raw, expected):
        assert _parse_binary_label(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("75", 75.0), ("52.5", 52.5), ("", None), ("x", None)],
    )
    def test_parse_float(self, raw, expected):
        assert _parse_float(raw) == expected


class TestManifestIntegrity:
    """對照實際資料的整合測試。"""

    @pytest.fixture(scope="class")
    def manifest(self):
        return build_manifest()

    def test_expected_subject_and_sample_count(self, manifest):
        subjects = {s.subject_id for s in manifest}
        assert len(subjects) == 31, "受試者人數與稽核結果不符"
        assert len(manifest) == 92, "動作樣本數與稽核結果不符"

    def test_label_distribution(self, manifest):
        labels = {s.subject_id: s.label for s in manifest}
        assert sum(labels.values()) == 12, "陽性人數與稽核結果不符"
        assert len(labels) - sum(labels.values()) == 19

    def test_excluded_subjects_absent(self, manifest):
        present = {s.subject_id for s in manifest}
        for uid in EXCLUDED_SUBJECTS:
            assert uid not in present

    def test_excluded_samples_absent(self, manifest):
        pairs = {(s.subject_id, s.action) for s in manifest}
        for pair in EXCLUDED_SAMPLES:
            assert pair not in pairs

    def test_pilot_batch_excluded(self, manifest):
        """P 系列為試作批，不應進入臨床模型。"""
        assert not any(s.subject_id.startswith("P") for s in manifest)

    def test_all_json_files_exist(self, manifest):
        missing = [s.json_path for s in manifest if not s.json_path.exists()]
        assert not missing, f"名冊指向不存在的檔案：{missing[:5]}"

    def test_subject_ids_are_uppercase(self, manifest):
        """a024 在原始資料為小寫，必須正規化，否則會被當成另一個人。"""
        for s in manifest:
            assert s.subject_id == s.subject_id.upper()

    def test_actions_are_known(self, manifest):
        assert {s.action for s in manifest} <= set(ACTIONS)

    def test_label_consistent_within_subject(self, manifest):
        seen = {}
        for s in manifest:
            if s.subject_id in seen:
                assert seen[s.subject_id] == s.label
            seen[s.subject_id] = s.label

    def test_profiles_within_plausible_range(self, manifest):
        """收案端已修正 A009 體重 525kg、A029 年齡 142 等鍵入錯誤。"""
        for s in manifest:
            assert 40 <= s.age <= 110, f"{s.subject_id} 年齡異常：{s.age}"
            assert 120 <= s.height_cm <= 200, f"{s.subject_id} 身高異常：{s.height_cm}"
            assert 25 <= s.weight_kg <= 150, f"{s.subject_id} 體重異常：{s.weight_kg}"
            assert 12 <= s.bmi <= 45, f"{s.subject_id} BMI 異常：{s.bmi:.1f}"

    def test_json_path_matches_subject_and_action(self, manifest):
        for s in manifest:
            stem = s.json_path.stem.lower()
            assert stem == f"{s.subject_id.lower()}_{s.action.lower()}"


class TestSubjectLoading:
    def test_unlabelled_subjects_dropped(self):
        """CSV 中標籤留白者不得進入名冊。"""
        subjects = load_subjects()
        assert all(s.label in (0, 1) for s in subjects.values())

    def test_fps_is_documented(self):
        assert FPS == 30.0

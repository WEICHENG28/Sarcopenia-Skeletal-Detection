"""交叉驗證切分的單元測試。

最重要的是資料洩漏檢查：同一位受試者若同時出現在訓練與驗證集，
模型可靠 static 特徵（年齡、BMI）認出他，驗證分數會虛高而無人察覺。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset_manifest import Sample
from src.splits import (
    check_no_leakage, iter_folds, make_folds, subject_labels,
)


def _fake_samples(n_pos: int = 12, n_neg: int = 19) -> list[Sample]:
    """造出與真實名冊相同結構的假樣本（31 人 × 3 動作）。"""
    samples = []
    for i in range(n_pos + n_neg):
        label = 1 if i < n_pos else 0
        uid = f"A{i:03d}"
        for action in ("Sit", "Walk", "Squat"):
            samples.append(
                Sample(
                    subject_id=uid,
                    action=action,
                    label=label,
                    age=70.0 + i,
                    height_cm=160.0,
                    weight_kg=60.0,
                    gender="female" if i % 2 else "male",
                    json_path=Path(f"/fake/{uid}_{action}.json"),
                )
            )
    return samples


class TestNoLeakage:
    def test_no_subject_in_both_sides(self):
        samples = _fake_samples()
        folds = make_folds(samples)
        check_no_leakage(folds)  # 有問題會直接拋錯

    def test_all_actions_of_a_subject_stay_together(self):
        """同一人的 Sit/Walk/Squat 必須整組落在同一側。"""
        samples = _fake_samples()
        for fold, train, val in iter_folds(samples):
            train_ids = {s.subject_id for s in train}
            val_ids = {s.subject_id for s in val}
            assert not (train_ids & val_ids)
            # 每個驗證集受試者的所有動作都應在驗證集內
            for uid in val_ids:
                assert all(s.subject_id != uid for s in train)

    def test_train_val_partition_is_complete(self):
        """訓練 + 驗證應恰好等於全部樣本，不重不漏。"""
        samples = _fake_samples()
        for fold, train, val in iter_folds(samples):
            assert len(train) + len(val) == len(samples)


class TestFoldStructure:
    def test_fold_count(self):
        folds = make_folds(_fake_samples(), n_splits=5, n_repeats=5)
        assert len(folds) == 25

    def test_each_subject_validated_once_per_repeat(self):
        """單次重複中，每人必須恰好被驗證一次，out-of-fold 才能涵蓋全體。"""
        samples = _fake_samples()
        folds = make_folds(samples, n_splits=5, n_repeats=3)
        all_ids = set(subject_labels(samples))
        for repeat in range(3):
            counts = {uid: 0 for uid in all_ids}
            for f in folds:
                if f.repeat != repeat:
                    continue
                for uid in f.val_subjects:
                    counts[uid] += 1
            assert set(counts.values()) == {1}, f"repeat {repeat} 未恰好覆蓋每人一次"

    def test_stratification_keeps_positives_in_every_fold(self):
        """每折驗證集都要有陽性樣本，否則該折算不出 AUC。"""
        samples = _fake_samples()
        labels = subject_labels(samples)
        for f in make_folds(samples, n_repeats=2):
            val_pos = sum(labels[u] for u in f.val_subjects)
            assert val_pos >= 1, f"repeat {f.repeat} fold {f.fold} 驗證集沒有陽性樣本"

    def test_repeats_produce_different_partitions(self):
        """不同重複必須真的產生不同切分，否則重複執行沒有意義。"""
        folds = make_folds(_fake_samples(), n_repeats=3)
        partitions = {
            r: frozenset(frozenset(f.val_subjects) for f in folds if f.repeat == r)
            for r in range(3)
        }
        assert partitions[0] != partitions[1]

    def test_deterministic_given_same_seed(self):
        a = make_folds(_fake_samples(), seed=123)
        b = make_folds(_fake_samples(), seed=123)
        assert [f.val_subjects for f in a] == [f.val_subjects for f in b]


class TestSubjectLabels:
    def test_collapses_actions_to_one_label(self):
        labels = subject_labels(_fake_samples())
        assert len(labels) == 31
        assert sum(labels.values()) == 12

    def test_rejects_contradictory_labels(self):
        samples = _fake_samples()
        bad = Sample(
            subject_id=samples[0].subject_id,
            action="Walk",
            label=1 - samples[0].label,
            age=70.0, height_cm=160.0, weight_kg=60.0,
            gender="male", json_path=Path("/fake/x.json"),
        )
        with pytest.raises(ValueError, match="矛盾"):
            subject_labels(samples + [bad])

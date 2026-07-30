"""評估工具：三個系統共用同一套指標計算，確保比較公平。

核心設計是 **out-of-fold（OOF）彙總**：
每位受試者在「輪到他當驗證集」那一折會得到一個預測值，
收齊全部 31 人後才算一次 AUC。

為什麼不平均每折的 AUC：每折驗證集僅 6~7 人（陽性 2~3），
單折 AUC 的取值離散且不穩定，平均起來會低估真實表現也高估穩定度。
折間分數仍會保留，但只用來呈現變異程度。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    recall_score, roc_auc_score,
)

# 醫療應用偏重不漏診（recall），因此門檻不固定在 0.5，
# 而是在訓練集上挑選使 F1 最大者。
DEFAULT_THRESHOLD = 0.5


@dataclass
class OOFCollector:
    """收集各折的 out-of-fold 預測，依受試者彙總。"""

    # {repeat: {subject_id: (機率, 標籤)}}
    _store: Dict[int, Dict[str, Tuple[float, int]]] = field(default_factory=lambda: defaultdict(dict))
    _fold_scores: List[float] = field(default_factory=list)

    def add(
        self,
        repeat: int,
        subjects: Sequence[str],
        probabilities: Sequence[float],
        labels: Sequence[int],
    ) -> None:
        """記錄一折的預測。同一人在同一次重複中只會出現一次。"""
        for subject, prob, label in zip(subjects, probabilities, labels):
            self._store[repeat][subject] = (float(prob), int(label))

        if len(set(labels)) > 1:
            self._fold_scores.append(float(roc_auc_score(labels, probabilities)))

    def repeat_aucs(self) -> List[float]:
        """每次重複的 OOF AUC（各自涵蓋全部受試者）。"""
        aucs = []
        for repeat in sorted(self._store):
            probs, labels = self._arrays(repeat)
            if len(set(labels)) > 1:
                aucs.append(float(roc_auc_score(labels, probs)))
        return aucs

    def _arrays(self, repeat: int) -> Tuple[np.ndarray, np.ndarray]:
        records = self._store[repeat]
        subjects = sorted(records)
        probs = np.array([records[s][0] for s in subjects])
        labels = np.array([records[s][1] for s in subjects])
        return probs, labels

    def mean_probabilities(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """跨重複平均每人的 OOF 機率，降低單次切分的隨機性。"""
        per_subject: Dict[str, List[float]] = defaultdict(list)
        label_of: Dict[str, int] = {}
        for repeat in self._store:
            for subject, (prob, label) in self._store[repeat].items():
                per_subject[subject].append(prob)
                label_of[subject] = label

        subjects = sorted(per_subject)
        probs = np.array([float(np.mean(per_subject[s])) for s in subjects])
        labels = np.array([label_of[s] for s in subjects])
        return subjects, probs, labels

    def summary(self) -> Dict[str, float]:
        """主要報告指標。"""
        subjects, probs, labels = self.mean_probabilities()
        repeat_aucs = self.repeat_aucs()

        if len(set(labels)) < 2:
            return {"n_subjects": float(len(subjects)), "auc": float("nan")}

        threshold = best_threshold(labels, probs)
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = _confusion(labels, preds)

        return {
            "n_subjects": float(len(subjects)),
            "auc": float(roc_auc_score(labels, probs)),
            "auc_std_across_repeats": float(np.std(repeat_aucs)) if repeat_aucs else 0.0,
            "auc_min_repeat": float(np.min(repeat_aucs)) if repeat_aucs else float("nan"),
            "auc_max_repeat": float(np.max(repeat_aucs)) if repeat_aucs else float("nan"),
            "pr_auc": float(average_precision_score(labels, probs)),
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, preds)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        }


def _confusion(labels: np.ndarray, preds: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tn, fp, fn, tp


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """挑選使 F1 最大的門檻。

    因為使用了類別權重，0.5 不再是合理的預設切點。
    """
    candidates = np.unique(probs)
    if len(candidates) < 2:
        return DEFAULT_THRESHOLD

    best, best_f1 = DEFAULT_THRESHOLD, -1.0
    for threshold in candidates:
        f1 = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best = f1, float(threshold)
    return best


def bootstrap_auc_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """AUC 的 95% 信賴區間（bootstrap）。

    31 人的樣本量下，這個區間會相當寬——這正是必須呈現的資訊，
    只報單一 AUC 會嚴重高估結論的確定性。
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        scores.append(roc_auc_score(labels[idx], probs[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def format_summary(name: str, summary: Dict[str, float], ci: Tuple[float, float] | None = None) -> str:
    """把指標排成一行，方便並排比較。"""
    auc = summary.get("auc", float("nan"))
    ci_text = f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci else ""
    return (
        f"{name:24s} AUC={auc:.3f}{ci_text}  "
        f"±{summary.get('auc_std_across_repeats', 0):.3f}  "
        f"PR-AUC={summary.get('pr_auc', float('nan')):.3f}  "
        f"recall={summary.get('recall', float('nan')):.3f}  "
        f"spec={summary.get('specificity', float('nan')):.3f}  "
        f"F1={summary.get('f1', float('nan')):.3f}"
    )

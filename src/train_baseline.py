"""系統 A：手工運動學特徵 + 傳統分類器。

這是三系統比較中的**對照基線**。在 31 人的樣本量下，
只有 5~15 個參數要學的簡單模型往往比數十萬參數的圖神經網路更穩定，
所以這個基線不是配角——它很可能是最終最可信的結果。

三個特徵分組（見 DEVELOPMENT_LOG「年齡混淆」）：
    demographic  只用年齡/BMI/身高/體重/性別 → 不看動作能做到多好
    motion       只用運動學特徵             → 動作本身有沒有訊號
    combined     兩者合併                   → 動作在年齡之上有無額外貢獻

執行：
    python -m src.train_baseline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset_manifest import ACTIONS, Sample, build_manifest
from src.evaluation import OOFCollector, bootstrap_auc_ci, format_summary
from src.features_handcrafted import (
    FEATURE_GROUPS, build_feature_matrix, select_feature_group,
)
from src.splits import make_folds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42


def make_classifier(kind: str) -> Pipeline:
    """建立分類器。

    ``class_weight="balanced"`` 處理 12:19 的類別不平衡；
    醫療應用漏診的代價高於誤報，這個設定會讓模型更願意判為陽性。
    """
    if kind == "logreg":
        model = LogisticRegression(
            C=0.5,               # 較強的正則化，31 人容易過擬合
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
    elif kind == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=3,         # 淺樹，避免記住個別受試者
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"未知的分類器：{kind}")

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", model),
    ])


@dataclass
class ActionResult:
    action: str
    group: str
    classifier: str
    summary: Dict[str, float]
    ci: Tuple[float, float]
    subject_probabilities: Dict[str, float]
    subject_labels: Dict[str, int]


def train_action(
    samples: Sequence[Sample],
    action: str,
    group: str,
    classifier: str,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> ActionResult:
    """對單一動作、單一特徵分組跑完整交叉驗證。"""
    matrix, labels, subjects, names = build_feature_matrix(samples, action)
    matrix, _ = select_feature_group(matrix, names, group)

    subject_index = {subject: i for i, subject in enumerate(subjects)}
    collector = OOFCollector()

    for fold in make_folds(samples, n_splits=n_splits, n_repeats=n_repeats):
        train_rows = [subject_index[s] for s in fold.train_subjects if s in subject_index]
        val_rows = [subject_index[s] for s in fold.val_subjects if s in subject_index]
        if not val_rows or len(set(labels[train_rows])) < 2:
            continue

        model = make_classifier(classifier)
        model.fit(matrix[train_rows], labels[train_rows])
        probs = model.predict_proba(matrix[val_rows])[:, 1]

        collector.add(
            repeat=fold.repeat,
            subjects=[subjects[i] for i in val_rows],
            probabilities=probs,
            labels=labels[val_rows],
        )

    summary = collector.summary()
    oof_subjects, oof_probs, oof_labels = collector.mean_probabilities()
    ci = bootstrap_auc_ci(oof_labels, oof_probs)

    return ActionResult(
        action=action,
        group=group,
        classifier=classifier,
        summary=summary,
        ci=ci,
        subject_probabilities=dict(zip(oof_subjects, oof_probs.tolist())),
        subject_labels=dict(zip(oof_subjects, oof_labels.tolist())),
    )


def fuse_actions(
    results: Sequence[ActionResult],
    weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Tuple[float, float], Dict[str, float]]:
    """Late fusion：以動態分母整合各動作的機率。

    分母只加總「該受試者實際完成」的動作權重，
    因此受試者僅完成 2/3 項動作時仍可產出分數（A003 即為真實案例）。
    """
    weights = weights or {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}

    numerator: Dict[str, float] = {}
    denominator: Dict[str, float] = {}
    labels: Dict[str, int] = {}

    for result in results:
        weight = weights[result.action]
        for subject, prob in result.subject_probabilities.items():
            numerator[subject] = numerator.get(subject, 0.0) + weight * prob
            denominator[subject] = denominator.get(subject, 0.0) + weight
            labels[subject] = result.subject_labels[subject]

    fused = {s: numerator[s] / denominator[s] for s in numerator if denominator[s] > 0}
    subjects = sorted(fused)
    probs = np.array([fused[s] for s in subjects])
    label_array = np.array([labels[s] for s in subjects])

    collector = OOFCollector()
    collector.add(repeat=0, subjects=subjects, probabilities=probs, labels=label_array)
    summary = collector.summary()
    ci = bootstrap_auc_ci(label_array, probs)
    return summary, ci, fused


def run(classifier: str = "logreg", n_repeats: int = 5) -> Dict[str, object]:
    """跑完整的系統 A 評估並印出結果。"""
    samples = build_manifest()
    payload: Dict[str, object] = {"classifier": classifier, "n_repeats": n_repeats, "groups": {}}

    print("=" * 96)
    print(f"系統 A：手工運動學特徵 + {classifier}")
    print(f"31 位受試者（陽性 12 / 陰性 19）  ·  5-fold × {n_repeats} 次重複  ·  以受試者分組")
    print("=" * 96)
    print("AUC 後方 [ ] 為 bootstrap 95% 信賴區間，± 為跨重複標準差")
    print()

    for group in FEATURE_GROUPS:
        print(f"--- 特徵分組：{group} ---")
        results: List[ActionResult] = []
        group_payload: Dict[str, object] = {}

        for action in ACTIONS:
            result = train_action(samples, action, group, classifier, n_repeats=n_repeats)
            results.append(result)
            group_payload[action] = {"summary": result.summary, "ci": list(result.ci)}
            print("  " + format_summary(action, result.summary, result.ci))

        fused_summary, fused_ci, _ = fuse_actions(results)
        group_payload["fusion"] = {"summary": fused_summary, "ci": list(fused_ci)}
        print("  " + format_summary("Late Fusion", fused_summary, fused_ci))
        print()

        payload["groups"][group] = group_payload

    _print_verdict(payload)
    return payload


def _print_verdict(payload: Dict[str, object]) -> None:
    """點出關鍵比較：動作特徵是否在人口學之上帶來額外資訊。"""
    groups = payload["groups"]
    demo = groups["demographic"]["fusion"]["summary"]["auc"]
    motion = groups["motion"]["fusion"]["summary"]["auc"]
    combined = groups["combined"]["fusion"]["summary"]["auc"]

    print("=" * 96)
    print("關鍵比較（Late Fusion AUC）")
    print("=" * 96)
    print(f"  只看人口學（年齡等）  {demo:.3f}   <- 不分析動作就能達到的基準")
    print(f"  只看動作特徵          {motion:.3f}")
    print(f"  兩者合併              {combined:.3f}")
    print()
    gain = combined - demo
    print(f"  動作特徵帶來的增益：{gain:+.3f}")
    if gain > 0.05:
        print("  -> 動作資料在人口學之上提供了額外資訊。")
    elif gain > 0.0:
        print("  -> 增益微弱，以 31 人的樣本量難以區分是真實效果或雜訊。")
    else:
        print("  -> 未觀察到增益。此結果本身即是重要發現，應如實報告。")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練系統 A（手工特徵基線）")
    parser.add_argument("--classifier", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="將結果寫入 results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(classifier=args.classifier, n_repeats=args.repeats)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"system_a_{args.classifier}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"結果已寫入：{out_path}")


if __name__ == "__main__":
    main()

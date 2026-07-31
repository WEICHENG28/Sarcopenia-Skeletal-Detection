"""動作組合消融：只做部分動作時，判斷力會掉多少？

用途是收案決策，回答三個問題：

1. 哪個動作最值得保留？（現場時間有限、長者體力有限）
2. 只能做兩項時該做哪兩項？
3. 刪掉某個動作的代價有多大？

方法：取各動作的 out-of-fold 預測機率，依動態分母 late fusion 重新組合，
對 7 種組合各算一次 AUC。所有組合共用同一批 OOF 預測，因此可直接比較。

執行：
    python -m src.ablation_actions
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.dataset_manifest import ACTIONS
from src.evaluation import bootstrap_auc_ci, OOFCollector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_WEIGHTS: Dict[str, float] = {"Sit": 0.4, "Walk": 0.4, "Squat": 0.2}


def fuse_subset(
    probabilities: Dict[str, Dict[str, float]],
    labels: Dict[str, Dict[str, int]],
    actions: Sequence[str],
    weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Tuple[float, float], int]:
    """只用指定動作做 late fusion。

    分母僅加總該受試者實際完成的動作權重，
    因此缺項受試者（如 A003 缺 Sit）仍可產出分數。
    """
    weights = weights or DEFAULT_WEIGHTS

    numerator: Dict[str, float] = {}
    denominator: Dict[str, float] = {}
    truth: Dict[str, int] = {}

    for action in actions:
        weight = weights[action]
        for subject, probability in probabilities[action].items():
            numerator[subject] = numerator.get(subject, 0.0) + weight * probability
            denominator[subject] = denominator.get(subject, 0.0) + weight
            truth[subject] = labels[action][subject]

    subjects = sorted(s for s in numerator if denominator[s] > 0)
    if not subjects:
        return {}, (float("nan"), float("nan")), 0

    fused = np.array([numerator[s] / denominator[s] for s in subjects])
    label_array = np.array([truth[s] for s in subjects])

    collector = OOFCollector()
    collector.add(repeat=0, subjects=subjects, probabilities=fused, labels=label_array)
    return collector.summary(), bootstrap_auc_ci(label_array, fused), len(subjects)


def load_oof(result_path: Path, group: str = "motion") -> Tuple[Dict, Dict]:
    """從訓練結果檔取出各動作的 out-of-fold 預測。"""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    section = payload["groups"][group]
    return section["_subject_probabilities"], section["_subject_labels"]


def run(result_path: Path, group: str = "motion") -> List[Dict[str, object]]:
    probabilities, labels = load_oof(result_path, group)

    subsets: List[Tuple[str, ...]] = []
    for size in (1, 2, 3):
        subsets.extend(combinations(ACTIONS, size))

    rows: List[Dict[str, object]] = []
    for subset in subsets:
        summary, ci, n_subjects = fuse_subset(probabilities, labels, subset)
        rows.append({
            "actions": list(subset),
            "n_actions": len(subset),
            "n_subjects": n_subjects,
            "summary": summary,
            "ci": list(ci),
        })
    return rows


def print_table(rows: List[Dict[str, object]], title: str) -> None:
    print("=" * 92)
    print(title)
    print("=" * 92)
    print(f"{'動作組合':26s} {'人數':>4s} {'AUC':>7s} {'95% CI':>15s} "
          f"{'recall':>7s} {'spec':>7s} {'F1':>7s}")
    print("-" * 92)

    full = next(r for r in rows if r["n_actions"] == 3)
    full_auc = full["summary"]["auc"]

    for size in (3, 2, 1):
        for row in sorted(
            (r for r in rows if r["n_actions"] == size),
            key=lambda r: r["summary"]["auc"],
            reverse=True,
        ):
            summary = row["summary"]
            name = " + ".join(row["actions"])
            delta = summary["auc"] - full_auc
            marker = "" if size == 3 else f"  ({delta:+.3f})"
            print(
                f"{name:26s} {row['n_subjects']:4d} {summary['auc']:7.3f} "
                f"  [{row['ci'][0]:.2f}-{row['ci'][1]:.2f}]  "
                f"{summary['recall']:7.3f} {summary['specificity']:7.3f} "
                f"{summary['f1']:7.3f}{marker}"
            )
        print()


def print_verdict(rows: List[Dict[str, object]]) -> None:
    """給收案端的具體建議。"""
    singles = {r["actions"][0]: r["summary"]["auc"] for r in rows if r["n_actions"] == 1}
    full = next(r for r in rows if r["n_actions"] == 3)["summary"]["auc"]
    pairs = {
        tuple(r["actions"]): r["summary"]["auc"] for r in rows if r["n_actions"] == 2
    }

    print("=" * 92)
    print("收案決策參考")
    print("=" * 92)

    best_single = max(singles, key=singles.get)
    print(f"  單項最強：{best_single}（AUC {singles[best_single]:.3f}）")

    best_pair = max(pairs, key=pairs.get)
    print(f"  兩項最佳組合：{' + '.join(best_pair)}（AUC {pairs[best_pair]:.3f}，"
          f"較三項全做 {pairs[best_pair] - full:+.3f}）")

    print()
    print("  刪掉單一動作的代價：")
    for action in ACTIONS:
        remaining = tuple(a for a in ACTIONS if a != action)
        cost = pairs[remaining] - full
        verdict = "可考慮省略" if cost >= -0.02 else "建議保留"
        print(f"    不做 {action:6s} -> AUC {pairs[remaining]:.3f}（{cost:+.3f}）  {verdict}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動作組合消融實驗")
    parser.add_argument(
        "--result", type=Path,
        default=RESULTS_DIR / "system_stgcn_pretrained.json",
        help="訓練結果 JSON 路徑",
    )
    parser.add_argument("--group", default="motion", choices=["motion", "combined"])
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.result.exists():
        raise FileNotFoundError(f"找不到結果檔：{args.result}")

    rows = run(args.result, args.group)
    print_table(rows, f"動作組合消融（{args.result.stem} · {args.group} 分組）")
    print_verdict(rows)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"ablation_{args.result.stem}_{args.group}.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"結果已寫入：{out_path}")


if __name__ == "__main__":
    main()

"""訓練名冊：把治療師標籤 CSV 與 data/A_*_json/ 底下的骨架檔對應起來。

名冊是整條管線的單一資料來源（single source of truth），
系統 A（手工特徵）、系統 B（ST-GCN）、系統 C（Shift-GCN）都從這裡取樣本，
確保三者跑在完全相同的資料上，比較結果才有意義。

排除規則全部寫在 EXCLUDED_SUBJECTS / EXCLUDED_SAMPLES，
每一筆都附理由，方便日後稽核「為什麼少了這幾筆」。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_CSV_PATH = PROJECT_ROOT / "目前檔案範例" / "patient_data(summer).csv"
DATA_DIR = PROJECT_ROOT / "data"

ACTIONS: Tuple[str, ...] = ("Sit", "Walk", "Squat")
ACTION_DIRS: Dict[str, str] = {
    "Sit": "A_sit_json",
    "Walk": "A_walk_json",
    "Squat": "A_squat_json",
}

# 影片幀率（全批統一 30fps，由收案團隊確認）
FPS = 30.0

# 整位受試者排除
EXCLUDED_SUBJECTS: Dict[str, str] = {
    "A001": "CSV 有標籤但缺骨架 JSON",
    "A002": "CSV 有標籤但缺骨架 JSON",
}

# 單筆動作樣本排除（受試者其餘動作仍保留）
EXCLUDED_SAMPLES: Dict[Tuple[str, str], str] = {
    ("A003", "Sit"): "visibility 僅 0.63（40% 幀 <0.5）、長度 40.5s 明顯偏離其他受試者，收案端確認先不使用",
}


@dataclass(frozen=True)
class Sample:
    """一筆「受試者 × 動作」樣本。"""

    subject_id: str
    action: str
    label: int  # 1 = 有肌少症, 0 = 無
    age: float
    height_cm: float
    weight_kg: float
    gender: str
    json_path: Path

    @property
    def bmi(self) -> float:
        h_m = self.height_cm / 100.0
        return self.weight_kg / (h_m * h_m)


@dataclass(frozen=True)
class Subject:
    """受試者層級的基本資料與標籤。"""

    subject_id: str
    label: int
    age: float
    height_cm: float
    weight_kg: float
    gender: str

    @property
    def bmi(self) -> float:
        h_m = self.height_cm / 100.0
        return self.weight_kg / (h_m * h_m)


def _parse_binary_label(raw: str) -> int | None:
    """把 '1' / '1.0' / '0' / '0.0' / '' 統一成 int 或 None。"""
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value in (0, 1) else None


def _parse_float(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_subjects(csv_path: Path = LABEL_CSV_PATH) -> Dict[str, Subject]:
    """讀標籤 CSV，回傳 {subject_id: Subject}。

    CSV 是 long format（每人每動作一列），同一人的基本資料會重複出現，
    這裡收斂成每人一筆並檢查是否有互相矛盾的基本資料。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"標籤 CSV 不存在：{csv_path}")

    # utf-8-sig：CSV 首欄名帶有 BOM
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    profiles: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        # a024 在 CSV 與檔名皆為小寫，統一大寫避免被當成另一個人
        uid = (row.get("user_id") or "").strip().upper()
        if uid:
            profiles.setdefault(uid, []).append(row)

    subjects: Dict[str, Subject] = {}
    for uid, uid_rows in profiles.items():
        labels = {_parse_binary_label(r.get("therapist_label") or "") for r in uid_rows}
        labels.discard(None)
        if len(labels) != 1:
            # 無標籤或標籤矛盾 → 不進訓練名冊
            continue

        age = _parse_float(uid_rows[0].get("age") or "")
        height = _parse_float(uid_rows[0].get("height_cm") or "")
        weight = _parse_float(uid_rows[0].get("weight_kg") or "")
        if age is None or height is None or weight is None:
            continue

        subjects[uid] = Subject(
            subject_id=uid,
            label=labels.pop(),
            age=age,
            height_cm=height,
            weight_kg=weight,
            gender=(uid_rows[0].get("gender") or "").strip().lower(),
        )

    return subjects


def _find_json(subject_id: str, action: str) -> Path | None:
    """在對應動作資料夾裡找骨架 JSON（檔名大小寫不敏感）。"""
    action_dir = DATA_DIR / ACTION_DIRS[action]
    if not action_dir.is_dir():
        return None

    target = f"{subject_id}_{action}.json".lower()
    for candidate in action_dir.glob("*.json"):
        if candidate.name.lower() == target:
            return candidate
    return None


def build_manifest(verbose: bool = False) -> List[Sample]:
    """組出訓練名冊，套用排除規則。"""
    subjects = load_subjects()
    samples: List[Sample] = []
    skipped: List[str] = []

    for uid in sorted(subjects):
        if uid in EXCLUDED_SUBJECTS:
            skipped.append(f"{uid}（整位排除）：{EXCLUDED_SUBJECTS[uid]}")
            continue

        subject = subjects[uid]
        for action in ACTIONS:
            reason = EXCLUDED_SAMPLES.get((uid, action))
            if reason is not None:
                skipped.append(f"{uid}/{action}：{reason}")
                continue

            json_path = _find_json(uid, action)
            if json_path is None:
                skipped.append(f"{uid}/{action}：找不到骨架 JSON")
                continue

            samples.append(
                Sample(
                    subject_id=uid,
                    action=action,
                    label=subject.label,
                    age=subject.age,
                    height_cm=subject.height_cm,
                    weight_kg=subject.weight_kg,
                    gender=subject.gender,
                    json_path=json_path,
                )
            )

    if verbose:
        for line in skipped:
            print(f"  [排除] {line}")

    return samples


def summarize(samples: List[Sample]) -> None:
    """印出名冊摘要，供人工核對。"""
    subject_labels: Dict[str, int] = {s.subject_id: s.label for s in samples}
    pos = sum(1 for v in subject_labels.values() if v == 1)
    neg = sum(1 for v in subject_labels.values() if v == 0)

    print(f"受試者：{len(subject_labels)} 人（有肌少症 {pos} / 無 {neg}）")
    print(f"動作樣本：{len(samples)} 筆")
    print()
    print("各動作樣本數：")
    for action in ACTIONS:
        action_samples = [s for s in samples if s.action == action]
        a_pos = sum(1 for s in action_samples if s.label == 1)
        print(f"  {action:6s}: {len(action_samples):3d} 筆（陽性 {a_pos}、陰性 {len(action_samples) - a_pos}）")

    print()
    incomplete = {
        uid: sorted(s.action for s in samples if s.subject_id == uid)
        for uid in subject_labels
    }
    partial = {uid: acts for uid, acts in incomplete.items() if len(acts) < len(ACTIONS)}
    if partial:
        print("動作不齊全的受試者：")
        for uid, acts in sorted(partial.items()):
            print(f"  {uid}: 只有 {', '.join(acts)}")
    else:
        print("所有受試者三動作皆齊全。")


if __name__ == "__main__":
    print("=== 排除紀錄 ===")
    manifest = build_manifest(verbose=True)
    print()
    print("=== 名冊摘要 ===")
    summarize(manifest)

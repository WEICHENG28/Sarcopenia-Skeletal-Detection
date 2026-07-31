"""合成（模擬）資料的名冊。

用途：**僅供預訓練**，讓 ST-GCN 先在較大的樣本上學會「動作長什麼樣」，
再以真實個案微調。真實資料只有 31 人，不足以從零學起。

## 為什麼只用骨架、不用人口學欄位

合成資料的年齡–標籤關係與真實資料**相反**：

| 資料 | 陰性平均年齡 | 陽性平均年齡 | 年齡 AUC |
|---|---|---|---|
| 真實 | 74.1 | 87.4 | 0.879（越老越可能陽性）|
| 合成 | 58.7 | 48.2 | 0.341（越年輕越可能陽性）|

身高也差 24 公分（合成 178.6 vs 真實 154.5），年齡下限 23 歲——
並非長照中心的高齡族群。因此人口學欄位一律不採用，
預訓練僅以骨架序列為輸入（motion-only）。

## 動作–標籤方向的驗證

以手工運動學特徵比對兩邊的判別方向：

| 動作 | 方向一致 | 方向相反 |
|---|---|---|
| Walk | 5 | 1 |
| Sit | 4 | 3 |
| Squat | 2 | 0 |

Walk 的關鍵特徵完全對齊（duration、gait_speed、step_period），
Sit 則有最強特徵 `knee_rom_deg` 方向相反的風險，需在結果中分開檢視。

另注意：合成資料的類別可分性遠高於真實（部分特徵 AUC 達 0.98），
代表模擬時把差異放大了。預訓練仍可學到有用的表徵，
但不應期待這個高分能轉移到真實資料。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from src.dataset_manifest import ACTIONS, Sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "真資料+假資料"
SYNTHETIC_CSV = SYNTHETIC_DIR / "patient_data.csv"

# 合成資料的動作資料夾命名與真實資料不同
SYNTHETIC_ACTION_DIRS: Dict[str, str] = {
    "Sit": "sit_json",
    "Walk": "walk",
    "Squat": "squat_json",
}

# 少於此大小的 JSON 視為損毀（實測 klv_*.json 僅 2 bytes）
_MIN_FILE_BYTES = 100


def _parse_label(raw: str) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value in (0, 1) else None


def build_synthetic_manifest(prefix: str = "S") -> List[Sample]:
    """組出合成資料的名冊。

    Args:
        prefix: 只採用此前綴的受試者。預設 "S"（模擬個案）；
            資料夾中另有少量 A/L/klv 開頭的檔案，前者與真實資料重複、
            後者為損毀檔，一律排除。
    """
    if not SYNTHETIC_CSV.exists():
        raise FileNotFoundError(f"合成資料 CSV 不存在：{SYNTHETIC_CSV}")

    with SYNTHETIC_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    profiles: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        uid = (row.get("user_id") or "").strip().upper()
        if uid.startswith(prefix.upper()):
            profiles.setdefault(uid, []).append(row)

    samples: List[Sample] = []
    for uid in sorted(profiles):
        labels = {_parse_label(r.get("therapist_label") or "") for r in profiles[uid]}
        labels.discard(None)
        if len(labels) != 1:
            continue
        label = labels.pop()

        row = profiles[uid][0]
        try:
            age = float(row["age"])
            height = float(row["height_cm"])
            weight = float(row["weight_kg"])
        except (KeyError, ValueError):
            continue

        for action in ACTIONS:
            path = SYNTHETIC_DIR / SYNTHETIC_ACTION_DIRS[action] / f"{uid}_{action}.json"
            if not path.exists() or path.stat().st_size < _MIN_FILE_BYTES:
                continue
            samples.append(
                Sample(
                    subject_id=uid,
                    action=action,
                    label=label,
                    age=age,
                    height_cm=height,
                    weight_kg=weight,
                    gender=(row.get("gender") or "").strip().lower(),
                    json_path=path,
                )
            )

    return samples


def summarize(samples: List[Sample]) -> None:
    labels = {s.subject_id: s.label for s in samples}
    positives = sum(labels.values())
    print(f"合成受試者：{len(labels)} 人（陽性 {positives} / 陰性 {len(labels) - positives}）")
    print(f"動作樣本：{len(samples)} 筆")
    for action in ACTIONS:
        count = sum(1 for s in samples if s.action == action)
        print(f"  {action:6s}: {count} 筆")


if __name__ == "__main__":
    summarize(build_synthetic_manifest())

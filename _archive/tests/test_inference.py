"""Self-test for Phase 4 inference module.

Run:
    python test_inference.py

重點：
- 本檔案示範如何在 payload 中明確串接 `.trc` 路徑。
- `Sit` 動作會放入「含 TRC」案例；`Walk` 則示範 `trc_path=None` 的降級情境。
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from src.inference import run_inference


def _pick_mot_with_trc_pair() -> tuple[str, str]:
    """Pick one MOT that has sibling TRC (explicit TRC demo case)."""
    mot_files = sorted(Path("data/raw").glob("*.mot"))
    if not mot_files:
        raise FileNotFoundError("No .mot files found under data/raw")

    for mot in mot_files:
        trc = mot.with_suffix(".trc")
        if trc.exists():
            return str(mot), str(trc)

    raise FileNotFoundError("No paired .mot/.trc files found under data/raw")


def _pick_mot_without_trc(fallback_mot: str) -> str:
    """Pick one MOT without sibling TRC; fallback to provided MOT if not found."""
    mot_files = sorted(Path("data/raw").glob("*.mot"))
    for mot in mot_files:
        if not mot.with_suffix(".trc").exists():
            return str(mot)
    return fallback_mot


def main() -> None:
    sit_mot, sit_trc = _pick_mot_with_trc_pair()
    walk_mot = _pick_mot_without_trc(fallback_mot=sit_mot)
    walk_json = "data/raw/P001_sit.json"

    payload = {
        "profile": {
            "age": 34,
            "height_cm": 172,
            "weight_kg": 74,
        },
        "actions": {
            "Sit": {
                "performed": True,
                "mot_path": sit_mot,
                # 交接重點：明確放入 .trc 路徑，示範 TRC 串接方式。
                "trc_path": sit_trc,
            },
            "Walk": {
                "performed": True,
                "mot_path": walk_mot,
                # 對照案例：不提供 TRC，驗證 fallback 仍可推論。
                "trc_path": None,
                # Shift-GCN 新介面：可用 json_path 直接提供 33 點骨架序列。
                "json_path": walk_json,
            },
            "Squat": {
                "performed": False,
                "mot_path": walk_mot,
            },
        },
    }

    result = run_inference(payload)

    print("=== Inference Result (含 TRC 串接示範) ===")
    print(f"Sit mock uses TRC path: {payload['actions']['Sit']['trc_path']}")
    print(f"Walk mock TRC fallback: {payload['actions']['Walk']['trc_path']}")
    print(f"OK: {result['ok']}")
    print(f"Message: {result['message']}")
    print(f"Final risk probability: {result['final_risk_probability']}")
    print(f"Used actions: {result.get('used_actions')}")
    print(f"Dynamic denominator: {result.get('dynamic_weight_denominator')}")
    print(f"Explanation mode: {result.get('explanation_mode')}")
    print("Top 3 explanation features:")

    for idx, item in enumerate(result.get("explanations_top3", []), start=1):
        print(
            f"  {idx}. {item['feature']} | direction={item['direction']} | "
            f"impact={item['impact']:.6f} | abs={item['abs_impact']:.6f}"
        )

    print("\nPer-action detail:")
    pprint(result.get("per_action", {}))


if __name__ == "__main__":
    main()


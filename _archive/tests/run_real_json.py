"""Run full real JSON inference pipeline.

Steps:
1) Read data/raw/real_test.json
2) Convert to bridge TRC/MOT (height_cm=165)
3) Build minimal registry CSV and run src/prepare_data.py in bridge mode
4) Run src/inference.py API (run_inference) and print final probability + top3 features
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.bridge_mediapipe import convert_mediapipe_json_to_motion_files
from src.inference import run_inference


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REAL_JSON_PATH = RAW_DIR / "real_test.json"
REGISTRY_PATH = PROJECT_ROOT / "data" / "raw_registry_real_test.csv"
PREPARED_OUTPUT_PATH = RAW_DIR / "database_real_test.csv"


def _convert_actions(user_id: str, height_cm: float) -> Dict[str, Dict[str, str]]:
    outputs: Dict[str, Dict[str, str]] = {}
    for action in ("Sit", "Walk", "Squat"):
        converted = convert_mediapipe_json_to_motion_files(
            json_path=REAL_JSON_PATH,
            patient_id=user_id,
            action_name=action,
            height_cm=height_cm,
            output_dir=RAW_DIR,
        )
        outputs[action] = converted
    return outputs


def _write_min_registry(
    *,
    registry_path: Path,
    user_id: str,
    age: float,
    gender: float,
    height_cm: float,
    weight_kg: float,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "User_ID",
        "Label",
        "age",
        "gender",
        "height_cm",
        "weight_kg",
    ]
    with registry_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "User_ID": user_id,
                "Label": "",
                "age": str(float(age)),
                "gender": str(float(gender)),
                "height_cm": str(float(height_cm)),
                "weight_kg": str(float(weight_kg)),
            }
        )


def _run_prepare_data(registry_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "prepare_data.py"),
        "--raw-registry",
        str(registry_path),
        "--output",
        str(output_path),
        "--bridge-mode",
    ]
    subprocess.run(cmd, check=True)


def _build_inference_payload(
    *,
    profile: Dict[str, Any],
    motion_files: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    actions: Dict[str, Dict[str, Any]] = {}
    for action in ("Sit", "Walk", "Squat"):
        paths = motion_files[action]
        actions[action] = {
            "performed": True,
            "mot_path": paths["mot_path"],
            "trc_path": paths["trc_path"],
        }
    return {"profile": profile, "actions": actions}


def _print_top3(top3: List[Dict[str, Any]]) -> None:
    print("Top 3 features:")
    for idx, item in enumerate(top3[:3], start=1):
        feature = item.get("feature")
        direction = item.get("direction")
        impact = float(item.get("impact", 0.0))
        print(f"{idx}. {feature} | {direction} | impact={impact:.8f}")


def main() -> None:
    if not REAL_JSON_PATH.exists():
        raise FileNotFoundError(f"Real JSON not found: {REAL_JSON_PATH}")

    user_id = "REALTEST"
    # Minimal profile assumptions for end-to-end run.
    profile = {
        "age": 65.0,
        "gender": 0.0,
        "height_cm": 165.0,
        "weight_kg": 60.0,
    }

    motion_files = _convert_actions(user_id=user_id, height_cm=165.0)
    _write_min_registry(
        registry_path=REGISTRY_PATH,
        user_id=user_id,
        age=profile["age"],
        gender=profile["gender"],
        height_cm=profile["height_cm"],
        weight_kg=profile["weight_kg"],
    )
    _run_prepare_data(registry_path=REGISTRY_PATH, output_path=PREPARED_OUTPUT_PATH)

    payload = _build_inference_payload(profile=profile, motion_files=motion_files)
    result = run_inference(payload)

    if not bool(result.get("ok")):
        raise RuntimeError(f"Inference failed: {result}")

    print(f"Final risk probability: {float(result['final_risk_probability']):.8f}")
    _print_top3(list(result.get("explanations_top3", [])))

    print("Generated motion files:")
    for action in ("Sit", "Walk", "Squat"):
        paths = motion_files[action]
        print(f"- {action} TRC: {paths['trc_path']}")
        print(f"- {action} MOT: {paths['mot_path']}")


if __name__ == "__main__":
    main()


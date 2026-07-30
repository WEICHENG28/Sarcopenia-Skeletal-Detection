"""單鏡頭閉環測試：JSON -> bridge(trc/mot) -> prepare -> train -> inference。

執行：
    python test_monocular_pipeline.py

重點：
1) 產生 mock MediaPipe Pose JSON（含 Sit/Walk）。
2) 呼叫 bridge 產出 [ID]_[Action].trc/.mot。
3) 產生 bridge 測試專用 registry。
4) 呼叫 prepare 產出 data/database.csv。
5) 呼叫 train 產出 phase4_cnn_*.pth + metadata。
6) 呼叫 inference 並印出最終機率與 Top3 解釋。
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.bridge_mediapipe import (
    MP_LEFT_ANKLE,
    MP_LEFT_FOOT_INDEX,
    MP_LEFT_HIP,
    MP_LEFT_KNEE,
    MP_LEFT_SHOULDER,
    MP_NOSE,
    MP_RIGHT_ANKLE,
    MP_RIGHT_FOOT_INDEX,
    MP_RIGHT_HIP,
    MP_RIGHT_KNEE,
    MP_RIGHT_SHOULDER,
    convert_mediapipe_json_to_motion_files,
)
from src.inference import run_inference


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REGISTRY_PATH = PROJECT_ROOT / "data" / "raw_registry_monocular_mock.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "database.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def _build_mock_frames(action: str, n_frames: int, variant: int) -> List[List[Dict[str, float]]]:
    """產生最小可用 33 keypoints 序列（MediaPipe 風格）。"""
    frames: List[List[Dict[str, float]]] = []
    for t in range(n_frames):
        phase = (2.0 * math.pi * t) / max(1, n_frames - 1)

        # 依使用者變體與動作注入差異，避免樣本過於同質
        body_shift_x = 0.01 * variant
        body_shift_z = 0.005 * variant

        if action.lower() == "sit":
            sit_depth = 0.14 * (0.5 - 0.5 * math.cos(phase))
            trunk_lean = 0.03 * math.sin(phase)
            l_step = 0.01 * math.sin(phase)
            r_step = 0.01 * math.sin(phase + math.pi)
        elif action.lower() == "walk":
            sit_depth = 0.03 * (0.5 - 0.5 * math.cos(phase))
            trunk_lean = 0.02 * math.sin(phase)
            l_step = 0.06 * math.sin(phase)
            r_step = 0.06 * math.sin(phase + math.pi)
        elif action.lower() == "squat":
            sit_depth = 0.18 * (0.5 - 0.5 * math.cos(phase))
            trunk_lean = 0.04 * math.sin(phase)
            l_step = 0.015 * math.sin(phase)
            r_step = 0.015 * math.sin(phase + math.pi)
        else:
            raise ValueError(f"Unsupported action: {action}")

        pts = [[0.0, 0.0, 0.0] for _ in range(33)]

        # torso
        pts[MP_NOSE] = [0.00 + body_shift_x, 1.65 - sit_depth * 0.3, 0.02 + body_shift_z]
        pts[MP_LEFT_SHOULDER] = [-0.18 + body_shift_x, 1.45 - sit_depth * 0.2, 0.02 + trunk_lean + body_shift_z]
        pts[MP_RIGHT_SHOULDER] = [0.18 + body_shift_x, 1.45 - sit_depth * 0.2, 0.02 + trunk_lean + body_shift_z]
        pts[MP_LEFT_HIP] = [-0.12 + body_shift_x, 1.00 - sit_depth, 0.00 + body_shift_z]
        pts[MP_RIGHT_HIP] = [0.12 + body_shift_x, 1.00 - sit_depth, 0.00 + body_shift_z]

        # legs: walk 交錯擺動 + sit 下沉
        pts[MP_LEFT_KNEE] = [-0.12 + body_shift_x, 0.56 - sit_depth * 0.7, 0.03 + l_step + body_shift_z]
        pts[MP_RIGHT_KNEE] = [0.12 + body_shift_x, 0.56 - sit_depth * 0.7, 0.03 + r_step + body_shift_z]
        pts[MP_LEFT_ANKLE] = [-0.12 + body_shift_x, 0.08, 0.02 + l_step * 0.8 + body_shift_z]
        pts[MP_RIGHT_ANKLE] = [0.12 + body_shift_x, 0.08, 0.02 + r_step * 0.8 + body_shift_z]
        pts[MP_LEFT_FOOT_INDEX] = [-0.12 + body_shift_x, 0.03, 0.20 + l_step + body_shift_z]
        pts[MP_RIGHT_FOOT_INDEX] = [0.12 + body_shift_x, 0.03, 0.20 + r_step + body_shift_z]

        frames.append([{"x": p[0], "y": p[1], "z": p[2]} for p in pts])

    return frames


def _run_cmd(args: List[str]) -> str:
    print(f"\n[RUN] {' '.join(args)}")
    cp = subprocess.run(args, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
    if cp.stdout.strip():
        print(cp.stdout)
    if cp.stderr.strip():
        print("[stderr]")
        print(cp.stderr)
    return cp.stdout + ("\n" + cp.stderr if cp.stderr else "")


def _write_registry(users: List[Dict[str, str]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["User_ID", "Label", "age", "gender", "height_cm", "weight_kg"],
        )
        writer.writeheader()
        for u in users:
            writer.writerow(u)


def main() -> None:
    python_exe = sys.executable
    print(f"Python executable: {python_exe}")

    # 每個 action 至少 8 筆（0/1 各 4）以通過 stratified train/val split。
    users = [
        {"User_ID": "M9001", "Label": "0", "age": "28", "gender": "male", "height_cm": "170", "weight_kg": "68"},
        {"User_ID": "M9002", "Label": "1", "age": "44", "gender": "female", "height_cm": "160", "weight_kg": "61"},
        {"User_ID": "M9003", "Label": "0", "age": "36", "gender": "male", "height_cm": "176", "weight_kg": "79"},
        {"User_ID": "M9004", "Label": "1", "age": "52", "gender": "female", "height_cm": "158", "weight_kg": "57"},
        {"User_ID": "M9005", "Label": "0", "age": "31", "gender": "female", "height_cm": "166", "weight_kg": "63"},
        {"User_ID": "M9006", "Label": "1", "age": "48", "gender": "male", "height_cm": "174", "weight_kg": "82"},
        {"User_ID": "M9007", "Label": "0", "age": "39", "gender": "male", "height_cm": "169", "weight_kg": "71"},
        {"User_ID": "M9008", "Label": "1", "age": "55", "gender": "female", "height_cm": "156", "weight_kg": "54"},
    ]

    # 1) bridge: 產生 Sit / Walk 的 trc/mot
    produced_motion_files: List[Path] = []
    for idx, u in enumerate(users, start=1):
        user_id = u["User_ID"]
        height_cm = float(u["height_cm"])
        for action in ("Sit", "Walk", "Squat"):
            frames = _build_mock_frames(action=action, n_frames=48, variant=idx)
            out = convert_mediapipe_json_to_motion_files(
                frames_data=frames,
                patient_id=user_id,
                action_name=action,
                height_cm=height_cm,
                output_dir=RAW_DIR,
                fps=30.0,
            )
            mot = Path(out["mot_path"])
            trc = Path(out["trc_path"])
            if not mot.exists() or not trc.exists():
                raise RuntimeError(f"Bridge output missing for {user_id}-{action}: {out}")
            produced_motion_files.extend([mot, trc])

    # 2) registry（bridge mode 專用，僅 base 欄位）
    _write_registry(users)
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    # 3) prepare
    _run_cmd(
        [
            python_exe,
            "src/prepare_data.py",
            "--raw-registry",
            str(REGISTRY_PATH),
            "--output",
            str(DATABASE_PATH),
            "--target-len",
            "64",
            "--bridge-mode",
        ]
    )
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"database.csv not generated: {DATABASE_PATH}")

    # 4) train
    train_output = _run_cmd([python_exe, "src/train.py"])
    if "Saved model:" not in train_output and "Saved metadata:" not in train_output:
        raise RuntimeError("Training output does not contain expected model/metadata save logs")

    model_files = sorted(MODELS_DIR.glob("phase4_cnn_*.pth"))
    metadata_path = MODELS_DIR / "phase4_cnn_metadata.json"
    required_models = [
        MODELS_DIR / "phase4_cnn_sit.pth",
        MODELS_DIR / "phase4_cnn_walk.pth",
        MODELS_DIR / "phase4_cnn_squat.pth",
    ]
    missing_models = [p for p in required_models if not p.exists()]
    if missing_models:
        raise FileNotFoundError(f"Missing trained model files: {missing_models}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata missing: {metadata_path}")

    # 5) inference（用同一批 bridge 產出的檔案）
    sample_user = users[0]
    sample_id = sample_user["User_ID"]
    payload = {
        "profile": {
            "age": int(float(sample_user["age"])),
            "height_cm": float(sample_user["height_cm"]),
            "weight_kg": float(sample_user["weight_kg"]),
            "gender": sample_user["gender"],
        },
        "actions": {
            "Sit": {
                "performed": True,
                "mot_path": str(RAW_DIR / f"{sample_id}_Sit.mot"),
                "trc_path": str(RAW_DIR / f"{sample_id}_Sit.trc"),
            },
            "Walk": {
                "performed": True,
                "mot_path": str(RAW_DIR / f"{sample_id}_Walk.mot"),
                "trc_path": str(RAW_DIR / f"{sample_id}_Walk.trc"),
            },
            "Squat": {
                "performed": True,
                "mot_path": str(RAW_DIR / f"{sample_id}_Squat.mot"),
                "trc_path": str(RAW_DIR / f"{sample_id}_Squat.trc"),
            },
        },
    }

    result = run_inference(payload)
    if not bool(result.get("ok")):
        raise RuntimeError(f"Inference failed: {result}")

    print("\n=== 單鏡頭閉環測試結果 ===")
    print(f"最終風險機率: {result.get('final_risk_probability')}")
    print("Top3 解釋特徵:")
    for i, item in enumerate(result.get("explanations_top3", []), start=1):
        print(
            f"  {i}. {item.get('feature')} | direction={item.get('direction')} | "
            f"impact={float(item.get('impact', 0.0)):.6f} | abs={float(item.get('abs_impact', 0.0)):.6f}"
        )

    # 6) 驗收摘要
    print("\n[驗收] bridge 輸出檔（節錄）:")
    for p in produced_motion_files[:6]:
        print(f"  - {p}")
    print(f"[驗收] 訓練模型數量: {len(model_files)}")
    print(f"[驗收] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()


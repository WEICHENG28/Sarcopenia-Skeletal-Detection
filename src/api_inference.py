"""Stateless API-oriented inference CLI.

Requirements covered:
- Parse CLI args: user_id/age/height_cm/weight_kg/gender.
- Do NOT read/write CSV registries/databases.
- Find `{user_id}_{action}.json` under `data/raw/`.
- Convert each found JSON to `.trc/.mot` via bridge_mediapipe.
- Build payload from CLI static profile + generated motion paths.
- Call Phase 5 inference directly and print JSON-only output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure absolute package imports (`src.*`) work when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge_mediapipe import convert_mediapipe_json_to_motion_files
from src.inference import run_inference


RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_ACTIONS: Tuple[str, ...] = ("Sit", "Walk", "Squat")


class JsonArgParser(argparse.ArgumentParser):
    """Arg parser that raises ValueError instead of printing non-JSON errors."""

    def error(self, message: str) -> None:  # pragma: no cover - argparse hook
        raise ValueError(message)


def _emit_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = JsonArgParser(add_help=True)
    parser.add_argument("--user_id", required=True, type=str)
    parser.add_argument("--age", required=True, type=float)
    parser.add_argument("--height_cm", required=True, type=float)
    parser.add_argument("--weight_kg", required=True, type=float)
    parser.add_argument("--gender", required=True, type=str)
    parser.add_argument(
        "--mock_when_missing",
        action="store_true",
        help="When no matching JSON is found, generate minimal mock JSON files for Sit/Walk/Squat.",
    )
    return parser.parse_args()


def _build_minimal_mock_frames(n_frames: int = 20) -> List[List[Dict[str, float]]]:
    """Create minimal 33-keypoint frames compatible with bridge input schema."""
    frames: List[List[Dict[str, float]]] = []
    for t in range(n_frames):
        ratio = t / max(1, (n_frames - 1))
        squat = 0.07 * (1.0 - abs(2.0 * ratio - 1.0))

        pts = [[0.0, 0.0, 0.0] for _ in range(33)]
        # torso
        pts[0] = [0.0, 1.65 - squat * 0.2, 0.02]
        pts[11] = [-0.18, 1.45 - squat * 0.15, 0.02]
        pts[12] = [0.18, 1.45 - squat * 0.15, 0.02]
        pts[23] = [-0.12, 1.00 - squat, 0.0]
        pts[24] = [0.12, 1.00 - squat, 0.0]
        # legs
        pts[25] = [-0.12, 0.56 - squat * 0.5, 0.02]
        pts[26] = [0.12, 0.56 - squat * 0.5, 0.02]
        pts[27] = [-0.12, 0.08, 0.01]
        pts[28] = [0.12, 0.08, 0.01]
        pts[31] = [-0.12, 0.03, 0.18]
        pts[32] = [0.12, 0.03, 0.18]

        frame = [{"x": p[0], "y": p[1], "z": p[2]} for p in pts]
        frames.append(frame)

    return frames


def _create_mock_jsons(user_id: str, raw_dir: Path) -> List[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = _build_minimal_mock_frames(n_frames=20)
    out: List[Path] = []
    for action in SUPPORTED_ACTIONS:
        p = raw_dir / f"{user_id}_{action}.json"
        p.write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
        out.append(p)
    return out


def _discover_action_jsons(user_id: str, raw_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for action in SUPPORTED_ACTIONS:
        candidate = raw_dir / f"{user_id}_{action}.json"
        if candidate.exists():
            found[action] = candidate
    return found


def _to_profile(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "age": float(args.age),
        "height_cm": float(args.height_cm),
        "weight_kg": float(args.weight_kg),
        "gender": args.gender,
    }


def run_api_inference(args: argparse.Namespace) -> Dict[str, Any]:
    profile = _to_profile(args)

    action_jsons = _discover_action_jsons(user_id=args.user_id, raw_dir=RAW_DIR)
    if not action_jsons and bool(args.mock_when_missing):
        _create_mock_jsons(user_id=args.user_id, raw_dir=RAW_DIR)
        action_jsons = _discover_action_jsons(user_id=args.user_id, raw_dir=RAW_DIR)

    if not action_jsons:
        return {
            "user_id": args.user_id,
            "risk_probability": None,
            "top_features": [],
            "used_actions": [],
            "status": "error",
            "message": f"No action JSON found under {RAW_DIR} for user_id={args.user_id}",
        }

    actions_payload: Dict[str, Dict[str, Any]] = {}
    converted_actions: List[str] = []
    for action, json_path in action_jsons.items():
        convert_out = convert_mediapipe_json_to_motion_files(
            json_path=str(json_path),
            patient_id=str(args.user_id),
            action_name=action,
            height_cm=float(args.height_cm),
            output_dir=str(RAW_DIR),
        )
        actions_payload[action] = {
            "performed": True,
            "trc_path": convert_out["trc_path"],
            "mot_path": convert_out["mot_path"],
        }
        converted_actions.append(action)

    inference_payload: Dict[str, Any] = {
        "profile": profile,
        "actions": actions_payload,
    }
    inference_result = run_inference(inference_payload)

    if not bool(inference_result.get("ok", False)):
        return {
            "user_id": args.user_id,
            "risk_probability": None,
            "top_features": [],
            "used_actions": list(inference_result.get("used_actions", [])),
            "status": "error",
            "message": str(inference_result.get("message", "inference_failed")),
        }

    top_features = [
        str(item.get("feature", ""))
        for item in list(inference_result.get("explanations_top3", []))[:3]
        if isinstance(item, dict) and str(item.get("feature", "")).strip()
    ]

    return {
        "user_id": args.user_id,
        "risk_probability": float(inference_result.get("final_risk_probability")),
        "top_features": top_features,
        "used_actions": list(inference_result.get("used_actions", converted_actions)),
        "status": "ok",
        "message": "inference_success",
    }


def main() -> None:
    try:
        args = _parse_args()
        result = run_api_inference(args)
    except Exception as exc:
        result = {
            "user_id": None,
            "risk_probability": None,
            "top_features": [],
            "used_actions": [],
            "status": "error",
            "message": str(exc),
        }
    _emit_json(result)


if __name__ == "__main__":
    main()


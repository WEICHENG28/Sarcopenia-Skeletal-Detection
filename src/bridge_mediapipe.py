"""Bridge MediaPipe Pose JSON sequences to `.trc` / `.mot` motion files.

This module is intentionally self-contained for Step B:
- Read MediaPipe Pose 33-keypoint sequences from JSON path or in-memory payload.
- Reconstruct/regularize 3D landmarks with simple physics-inspired constraints.
- Export `.trc` (3D keypoints over time) and `.mot` (IK-like angle trajectories).

Output contract compatibility:
- Files may include headers.
- Must include parseable numeric rows (float-only tokens per row).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import json
import math

import numpy as np


# MediaPipe Pose landmark indices (33 keypoints)
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert to float: {value!r}") from exc


def _angle_deg(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _extract_frame_landmarks(frame: Any) -> np.ndarray:
    """Extract one frame as shape (33, 3) float array.

    Supported per-frame payload styles:
    1) list with 33 elements, each is {x,y,z} or [x,y,z]
    2) dict containing one of keys: landmarks, pose_landmarks, keypoints
       where value follows style (1)
    """
    if isinstance(frame, dict):
        for key in ("landmarks", "pose_landmarks", "keypoints"):
            if key in frame:
                frame = frame[key]
                break

    if not isinstance(frame, Sequence) or len(frame) < 33:
        raise ValueError("Each frame must provide at least 33 landmarks")

    coords: List[List[float]] = []
    for i in range(33):
        kp = frame[i]
        if isinstance(kp, dict):
            x = _safe_float(kp.get("x", 0.0))
            y = _safe_float(kp.get("y", 0.0))
            z = _safe_float(kp.get("z", 0.0))
        elif isinstance(kp, Sequence) and len(kp) >= 3:
            x = _safe_float(kp[0])
            y = _safe_float(kp[1])
            z = _safe_float(kp[2])
        else:
            raise ValueError(f"Invalid keypoint at index {i}: {kp!r}")
        coords.append([x, y, z])

    return np.asarray(coords, dtype=np.float64)


def _load_frames(json_path: str | Path | None, frames_data: Any | None) -> np.ndarray:
    """Load MediaPipe frames to ndarray with shape (T, 33, 3)."""
    payload: Any
    if json_path is not None:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif frames_data is not None:
        payload = frames_data
    else:
        raise ValueError("Either json_path or frames_data must be provided")

    if isinstance(payload, dict):
        for key in ("frames", "results", "data"):
            if key in payload and isinstance(payload[key], Sequence):
                payload = payload[key]
                break

    if not isinstance(payload, Sequence) or len(payload) == 0:
        raise ValueError("Input payload must be a non-empty frame sequence")

    frames = [_extract_frame_landmarks(frame) for frame in payload]
    return np.stack(frames, axis=0)


@dataclass
class LowPassFilter:
    y: float = 0.0
    a: float = 1.0
    initialized: bool = False

    def apply(self, x: float, alpha: float) -> float:
        if not self.initialized:
            self.y = x
            self.initialized = True
            return x
        self.a = alpha
        self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


@dataclass
class OneEuroFilter:
    freq: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self.x_prev: float | None = None
        self.dx_filter = LowPassFilter()
        self.x_filter = LowPassFilter()

    @staticmethod
    def _alpha(freq: float, cutoff: float) -> float:
        te = 1.0 / max(freq, 1e-8)
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-8))
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return self.x_filter.apply(x, 1.0)

        dx = (x - self.x_prev) * self.freq
        self.x_prev = x

        a_d = self._alpha(self.freq, self.d_cutoff)
        dx_hat = self.dx_filter.apply(dx, a_d)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a_x = self._alpha(self.freq, cutoff)
        return self.x_filter.apply(x, a_x)


def _apply_one_euro_filter(
    kpts: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Filter all keypoints with OneEuro per coordinate channel."""
    t, n, c = kpts.shape
    out = np.zeros_like(kpts)
    filters: List[List[OneEuroFilter]] = [
        [OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(c)]
        for _ in range(n)
    ]

    for i in range(t):
        for j in range(n):
            for d in range(c):
                out[i, j, d] = filters[j][d](float(kpts[i, j, d]))
    return out


def _anthropometric_scale_to_height(kpts: np.ndarray, height_cm: float) -> np.ndarray:
    """Scale normalized-ish coordinates to metric using subject height."""
    height_m = float(height_cm) / 100.0
    if height_m <= 0:
        raise ValueError("height_cm must be > 0")

    # Approximate body height from nose to mean ankle over time, robust via median.
    nose = kpts[:, MP_NOSE, :]
    mean_ankle = 0.5 * (kpts[:, MP_LEFT_ANKLE, :] + kpts[:, MP_RIGHT_ANKLE, :])
    est_heights = np.linalg.norm(nose - mean_ankle, axis=1)
    est = float(np.median(est_heights))
    est = max(est, 1e-6)
    scale = height_m / est
    return kpts * scale


def _ground_calibration(kpts: np.ndarray) -> np.ndarray:
    """Set ground plane so minimum bilateral ankle Y equals 0."""
    out = kpts.copy()
    ankles_y = np.concatenate(
        [out[:, MP_LEFT_ANKLE, 1], out[:, MP_RIGHT_ANKLE, 1]],
        axis=0,
    )
    min_y = float(np.min(ankles_y))
    out[:, :, 1] -= min_y
    return out


def _enforce_segment_length(proximal: np.ndarray, distal: np.ndarray, target_length: float) -> np.ndarray:
    vec = distal - proximal
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        # Stable fallback direction: +Y
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        norm = 1.0
    return proximal + vec / norm * target_length


def _bone_length_anchoring(kpts: np.ndarray, anchor_frames: int = 10) -> np.ndarray:
    """Anchor thigh/shank/trunk lengths from first N-frame averages."""
    out = kpts.copy()
    n_anchor = int(max(1, min(anchor_frames, out.shape[0])))
    ref = out[:n_anchor]

    def avg_len(a: int, b: int) -> float:
        return float(np.mean(np.linalg.norm(ref[:, a, :] - ref[:, b, :], axis=1)))

    target = {
        "l_thigh": avg_len(MP_LEFT_HIP, MP_LEFT_KNEE),
        "r_thigh": avg_len(MP_RIGHT_HIP, MP_RIGHT_KNEE),
        "l_shank": avg_len(MP_LEFT_KNEE, MP_LEFT_ANKLE),
        "r_shank": avg_len(MP_RIGHT_KNEE, MP_RIGHT_ANKLE),
        "trunk": float(
            np.mean(
                np.linalg.norm(
                    0.5 * (ref[:, MP_LEFT_SHOULDER, :] + ref[:, MP_RIGHT_SHOULDER, :])
                    - 0.5 * (ref[:, MP_LEFT_HIP, :] + ref[:, MP_RIGHT_HIP, :]),
                    axis=1,
                )
            )
        ),
    }

    for t in range(out.shape[0]):
        # Legs (kinematic chain: hip -> knee -> ankle)
        out[t, MP_LEFT_KNEE, :] = _enforce_segment_length(
            out[t, MP_LEFT_HIP, :], out[t, MP_LEFT_KNEE, :], target["l_thigh"]
        )
        out[t, MP_LEFT_ANKLE, :] = _enforce_segment_length(
            out[t, MP_LEFT_KNEE, :], out[t, MP_LEFT_ANKLE, :], target["l_shank"]
        )
        out[t, MP_RIGHT_KNEE, :] = _enforce_segment_length(
            out[t, MP_RIGHT_HIP, :], out[t, MP_RIGHT_KNEE, :], target["r_thigh"]
        )
        out[t, MP_RIGHT_ANKLE, :] = _enforce_segment_length(
            out[t, MP_RIGHT_KNEE, :], out[t, MP_RIGHT_ANKLE, :], target["r_shank"]
        )

        # Trunk length via shifting shoulder pair by same delta.
        mid_hip = 0.5 * (out[t, MP_LEFT_HIP, :] + out[t, MP_RIGHT_HIP, :])
        mid_shoulder = 0.5 * (out[t, MP_LEFT_SHOULDER, :] + out[t, MP_RIGHT_SHOULDER, :])
        desired_mid_shoulder = _enforce_segment_length(mid_hip, mid_shoulder, target["trunk"])
        delta = desired_mid_shoulder - mid_shoulder
        out[t, MP_LEFT_SHOULDER, :] += delta
        out[t, MP_RIGHT_SHOULDER, :] += delta

    return out


def _foot_locking(kpts: np.ndarray, fps: float = 30.0, speed_threshold: float = 0.03) -> np.ndarray:
    """Lock foot Y/Z when foot speed is low (stance phase heuristic)."""
    out = kpts.copy()
    feet = (MP_LEFT_ANKLE, MP_RIGHT_ANKLE)

    for foot_idx in feet:
        lock_yz: np.ndarray | None = None
        for t in range(1, out.shape[0]):
            v = (out[t, foot_idx, :] - out[t - 1, foot_idx, :]) * fps
            speed = float(np.linalg.norm(v))
            if speed < speed_threshold:
                if lock_yz is None:
                    lock_yz = out[t, foot_idx, 1:3].copy()
                out[t, foot_idx, 1:3] = lock_yz
            else:
                lock_yz = None

    return out


def _compute_mot_angles(kpts: np.ndarray) -> np.ndarray:
    """Compute per-frame [hip_flex, knee_flex, ankle_angle, trunk_lean] in degrees."""
    rows: List[List[float]] = []
    global_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for t in range(kpts.shape[0]):
        p = kpts[t]
        mid_hip = 0.5 * (p[MP_LEFT_HIP] + p[MP_RIGHT_HIP])
        mid_shoulder = 0.5 * (p[MP_LEFT_SHOULDER] + p[MP_RIGHT_SHOULDER])
        trunk_up = mid_shoulder - mid_hip

        # Hip flexion (left/right average): 180 - angle(trunk_up, thigh_down)
        l_thigh = p[MP_LEFT_KNEE] - p[MP_LEFT_HIP]
        r_thigh = p[MP_RIGHT_KNEE] - p[MP_RIGHT_HIP]
        l_hip_flex = 180.0 - _angle_deg(trunk_up, l_thigh)
        r_hip_flex = 180.0 - _angle_deg(trunk_up, r_thigh)
        hip_flex = 0.5 * (l_hip_flex + r_hip_flex)

        # Knee flexion (left/right average): 180 - angle(hip-knee, ankle-knee)
        l_knee_raw = _angle_deg(p[MP_LEFT_HIP] - p[MP_LEFT_KNEE], p[MP_LEFT_ANKLE] - p[MP_LEFT_KNEE])
        r_knee_raw = _angle_deg(p[MP_RIGHT_HIP] - p[MP_RIGHT_KNEE], p[MP_RIGHT_ANKLE] - p[MP_RIGHT_KNEE])
        knee_flex = 0.5 * ((180.0 - l_knee_raw) + (180.0 - r_knee_raw))

        # Ankle angle (left/right average): angle(shank_down, foot_dir)
        l_ank = _angle_deg(p[MP_LEFT_KNEE] - p[MP_LEFT_ANKLE], p[MP_LEFT_FOOT_INDEX] - p[MP_LEFT_ANKLE])
        r_ank = _angle_deg(p[MP_RIGHT_KNEE] - p[MP_RIGHT_ANKLE], p[MP_RIGHT_FOOT_INDEX] - p[MP_RIGHT_ANKLE])
        ankle_angle = 0.5 * (l_ank + r_ank)

        # Trunk lean vs global vertical
        trunk_lean = _angle_deg(trunk_up, global_up)

        rows.append([hip_flex, knee_flex, ankle_angle, trunk_lean])

    return np.asarray(rows, dtype=np.float64)


def _write_trc(path: Path, kpts: np.ndarray, fps: float = 30.0) -> None:
    """Write TRC-like file with numeric rows parseable by downstream pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tBridgeMediaPipe\n")
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\n")
        f.write(f"{fps:.6f}\t{fps:.6f}\t{kpts.shape[0]}\t33\n")

        for i in range(kpts.shape[0]):
            t = i / fps
            flat = kpts[i].reshape(-1)
            values = [float(i + 1), float(t)] + [float(x) for x in flat]
            f.write(" ".join(f"{v:.8f}" for v in values) + "\n")


def _write_mot(path: Path, mot_angles: np.ndarray, fps: float = 30.0) -> None:
    """Write MOT file with numeric rows parseable by downstream pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fps <= 0:
        raise ValueError("fps must be > 0")

    n_rows = int(mot_angles.shape[0])
    dt = 1.0 / float(fps)
    t_end = (n_rows - 1) * dt if n_rows > 0 else 0.0
    column_names = ["time", "HipFlexion", "KneeFlexion", "AnkleAngle", "TrunkLean"]

    with path.open("w", encoding="utf-8") as f:
        f.write("name BridgeMediaPipeIK\n")
        f.write(f"datacolumns {len(column_names)}\n")
        f.write(f"datarows {n_rows}\n")
        f.write(f"range 0 {t_end:.8f}\n")
        f.write("endheader\n")
        f.write("\t".join(column_names) + "\n")

        for i in range(n_rows):
            t = i * dt
            hip_flex, knee_flex, ankle_angle, trunk_lean = mot_angles[i]
            row = [t, hip_flex, knee_flex, ankle_angle, trunk_lean]
            f.write(" ".join(f"{float(v):.8f}" for v in row) + "\n")


def convert_mediapipe_json_to_motion_files(
    *,
    json_path: str | Path | None = None,
    frames_data: Any | None = None,
    patient_id: str,
    action_name: str,
    height_cm: float,
    output_dir: str | Path,
    fps: float = 30.0,
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.02,
    one_euro_d_cutoff: float = 1.0,
    foot_speed_threshold: float = 0.03,
) -> Dict[str, str]:
    """Convert MediaPipe Pose frames to `.trc` and `.mot` files.

    Args:
        json_path: JSON file path. Optional if `frames_data` is provided.
        frames_data: In-memory dict/list payload. Optional if `json_path` is provided.
        patient_id: Prefix for output filename.
        action_name: Action suffix for output filename.
        height_cm: Subject height for anthropometric scaling.
        output_dir: Destination folder.
        fps: Sequence frame rate.

    Returns:
        Dict with `trc_path` and `mot_path`.
    """
    if not str(patient_id).strip():
        raise ValueError("patient_id must be non-empty")
    if not str(action_name).strip():
        raise ValueError("action_name must be non-empty")

    kpts = _load_frames(json_path=json_path, frames_data=frames_data)
    kpts = _anthropometric_scale_to_height(kpts, height_cm=height_cm)
    kpts = _ground_calibration(kpts)
    kpts = _bone_length_anchoring(kpts, anchor_frames=10)
    kpts = _foot_locking(kpts, fps=fps, speed_threshold=foot_speed_threshold)
    kpts = _apply_one_euro_filter(
        kpts,
        fps=fps,
        min_cutoff=one_euro_min_cutoff,
        beta=one_euro_beta,
        d_cutoff=one_euro_d_cutoff,
    )

    mot_angles = _compute_mot_angles(kpts)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{patient_id}_{action_name}"
    trc_path = out_dir / f"{basename}.trc"
    mot_path = out_dir / f"{basename}.mot"

    _write_trc(trc_path, kpts, fps=fps)
    _write_mot(mot_path, mot_angles, fps=fps)

    return {
        "trc_path": str(trc_path),
        "mot_path": str(mot_path),
    }


def _iter_numeric_rows(lines: Iterable[str]) -> Iterable[List[float]]:
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if not parts:
            continue
        try:
            yield [float(p) for p in parts]
        except ValueError:
            continue


def _build_mock_frames(n_frames: int = 24) -> List[List[Dict[str, float]]]:
    """Build tiny synthetic MediaPipe-like sequence for minimal self-test."""
    frames: List[List[Dict[str, float]]] = []
    for t in range(n_frames):
        phase = (2.0 * math.pi * t) / max(1, n_frames - 1)
        squat = 0.08 * (0.5 - 0.5 * math.cos(phase))
        knee_shift = 0.02 * math.sin(phase)

        # initialize neutral 33 keypoints
        pts = [[0.0, 0.0, 0.0] for _ in range(33)]

        # torso
        pts[MP_NOSE] = [0.0, 1.65 - squat * 0.3, 0.02]
        pts[MP_LEFT_SHOULDER] = [-0.18, 1.45 - squat * 0.25, 0.02]
        pts[MP_RIGHT_SHOULDER] = [0.18, 1.45 - squat * 0.25, 0.02]
        pts[MP_LEFT_HIP] = [-0.12, 1.00 - squat, 0.00]
        pts[MP_RIGHT_HIP] = [0.12, 1.00 - squat, 0.00]

        # legs
        pts[MP_LEFT_KNEE] = [-0.12, 0.56 - squat * 0.6, 0.02 + knee_shift]
        pts[MP_RIGHT_KNEE] = [0.12, 0.56 - squat * 0.6, 0.02 + knee_shift]
        pts[MP_LEFT_ANKLE] = [-0.12, 0.08, 0.01]
        pts[MP_RIGHT_ANKLE] = [0.12, 0.08, 0.01]
        pts[MP_LEFT_FOOT_INDEX] = [-0.12, 0.03, 0.18]
        pts[MP_RIGHT_FOOT_INDEX] = [0.12, 0.03, 0.18]

        frame = [{"x": p[0], "y": p[1], "z": p[2]} for p in pts]
        frames.append(frame)
    return frames


def _run_minimal_self_test() -> Tuple[str, str]:
    """Run minimal self-test: write files and verify numeric rows exist."""
    out = convert_mediapipe_json_to_motion_files(
        frames_data=_build_mock_frames(n_frames=20),
        patient_id="99999",
        action_name="mock",
        height_cm=170.0,
        output_dir="data/raw",
    )

    trc_p = Path(out["trc_path"])
    mot_p = Path(out["mot_path"])
    if not trc_p.exists() or not mot_p.exists():
        raise RuntimeError("Self-test failed: output files were not created")

    trc_rows = list(_iter_numeric_rows(trc_p.read_text(encoding="utf-8").splitlines()))
    mot_rows = list(_iter_numeric_rows(mot_p.read_text(encoding="utf-8").splitlines()))
    if len(trc_rows) == 0 or len(mot_rows) == 0:
        raise RuntimeError("Self-test failed: numeric rows missing in TRC/MOT")

    return str(trc_p), str(mot_p)


if __name__ == "__main__":
    trc_file, mot_file = _run_minimal_self_test()
    print(f"Self-test passed. TRC: {trc_file}")
    print(f"Self-test passed. MOT: {mot_file}")

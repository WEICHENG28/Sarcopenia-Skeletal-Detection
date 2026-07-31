"""手工運動學特徵：把治療師的臨床觀察指標自動算出來（系統 A）。

設計原則：

1. **對應臨床意義**
   每個特徵都要能用一句話跟治療師解釋。這是系統 A 相對深度學習的核心價值——
   Logistic Regression 的係數可以直接跟臨床常識核對。

2. **角度優先於座標**
   關節角度不受平移、旋轉、身材尺度影響，比原始座標穩健得多。

3. **可見度加權**
   側面拍攝時被遮擋側的關節座標是 MediaPipe 推測出來的。
   雙側指標一律以 visibility 加權，避免採到假座標。

座標前提：輸入已經過 :func:`skeleton_io.normalize_sequence`，
因此 +Y 朝上、+X 朝受試者左側、+Z 朝前，且原點在髖部中點。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.dataset_manifest import FPS, Sample
from src.skeleton_io import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
    load_landmarks, normalize_sequence,
)

_EPS = 1e-8

# 站直時的膝角門檻（度）。低於此值視為處於下蹲/坐姿階段。
STANDING_KNEE_ANGLE = 155.0


def _angle_deg(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> np.ndarray:
    """逐幀計算 a-vertex-c 的夾角（度）。輸入 shape 皆為 (T, 3)。"""
    v1 = a - vertex
    v2 = c - vertex
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / np.maximum(n1 * n2, _EPS)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _smooth(signal: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Savitzky-Golay 平滑，抑制 MediaPipe 的逐幀抖動但保留波形轉折。"""
    window = int(fps * 0.4)  # 約 0.4 秒
    window = max(5, window | 1)  # 必須為奇數
    if len(signal) <= window:
        return signal
    return savgol_filter(signal, window_length=window, polyorder=2)


def _bilateral(
    left: np.ndarray,
    right: np.ndarray,
    left_vis: np.ndarray,
    right_vis: np.ndarray,
) -> np.ndarray:
    """以 visibility 加權合併雙側訊號，避免採用被遮擋側的推測座標。"""
    wl = np.maximum(left_vis, _EPS)
    wr = np.maximum(right_vis, _EPS)
    return (left * wl + right * wr) / (wl + wr)


def _knee_angles(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """雙側膝角（可見度加權）。站直約 170~180 度，深蹲約 70~90 度。"""
    left = _angle_deg(xyz[:, L_HIP], xyz[:, L_KNEE], xyz[:, L_ANKLE])
    right = _angle_deg(xyz[:, R_HIP], xyz[:, R_KNEE], xyz[:, R_ANKLE])
    lv = np.minimum.reduce([vis[:, L_HIP], vis[:, L_KNEE], vis[:, L_ANKLE]])
    rv = np.minimum.reduce([vis[:, R_HIP], vis[:, R_KNEE], vis[:, R_ANKLE]])
    return _bilateral(left, right, lv, rv)


def _trunk_lean(xyz: np.ndarray) -> np.ndarray:
    """軀幹相對鉛直方向的前傾角（度）。起立時越吃力通常越需要前傾借力。"""
    mid_shoulder = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])
    mid_hip = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
    trunk = mid_shoulder - mid_hip
    vertical = np.zeros_like(trunk)
    vertical[:, 1] = 1.0
    cos = np.sum(trunk * vertical, axis=-1) / np.maximum(
        np.linalg.norm(trunk, axis=-1), _EPS
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _vertical_extent(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """髖部到腳踝的鉛直距離。站立時大、坐下時小，用來偵測起立坐下週期。

    world landmarks 把髖中點固定在原點，所以量的是腳踝相對髖部的下沉量。
    """
    left = -xyz[:, L_ANKLE, 1]
    right = -xyz[:, R_ANKLE, 1]
    return _bilateral(left, right, vis[:, L_ANKLE], vis[:, R_ANKLE])


def _count_cycles(signal: np.ndarray, fps: float = FPS) -> Tuple[int, float, float]:
    """從週期性訊號數出重複次數。

    Returns:
        (次數, 平均週期秒數, 週期秒數的標準差)
        標準差反映節奏穩定度——疲勞或肌力不足時，後面幾次通常會變慢。
    """
    smoothed = _smooth(signal, fps)
    span = float(np.ptp(smoothed))
    if span < _EPS or len(smoothed) < 3:
        return 0, 0.0, 0.0

    peaks, _ = find_peaks(
        smoothed,
        prominence=span * 0.25,       # 只認明顯的起伏，濾掉小抖動
        distance=max(1, int(fps * 0.7)),  # 兩次動作至少間隔 0.7 秒
    )
    if len(peaks) < 2:
        return int(len(peaks)), 0.0, 0.0

    intervals = np.diff(peaks) / fps
    return int(len(peaks)), float(intervals.mean()), float(intervals.std())


def _static_features(sample: Sample) -> Dict[str, float]:
    """受試者基本資料。三個動作共用。"""
    return {
        "age": sample.age,
        "bmi": sample.bmi,
        "height_cm": sample.height_cm,
        "weight_kg": sample.weight_kg,
        "is_female": 1.0 if sample.gender.startswith("f") else 0.0,
    }


def _quality_features(vis: np.ndarray) -> Dict[str, float]:
    """資料品質指標。低可見度樣本的其他特徵可信度較低，讓模型知道這件事。"""
    lower_body = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    return {
        "vis_mean": float(vis.mean()),
        "vis_lower_body": float(vis[:, lower_body].mean()),
    }


def sit_features(xyz: np.ndarray, vis: np.ndarray) -> Dict[str, float]:
    """起立坐下（five-times sit-to-stand，AWGS 2019 體能表現項目）。"""
    duration = len(xyz) / FPS
    knee = _knee_angles(xyz, vis)
    lean = _trunk_lean(xyz)
    extent = _vertical_extent(xyz, vis)

    reps, cycle_mean, cycle_std = _count_cycles(extent)

    # 站立階段佔比：肌力不足者停留在起身過程的時間較長
    standing_ratio = float((knee > STANDING_KNEE_ANGLE).mean())

    # 起身速度：鉛直伸展量的最大變化率
    rise_rate = float(np.abs(np.diff(_smooth(extent))).max() * FPS) if len(extent) > 1 else 0.0

    return {
        "duration_s": duration,
        "reps": float(reps),
        "cycle_mean_s": cycle_mean,
        "cycle_std_s": cycle_std,
        "sec_per_rep": duration / reps if reps > 0 else duration,
        "knee_min_deg": float(knee.min()),
        "knee_rom_deg": float(np.ptp(knee)),
        "trunk_lean_max_deg": float(lean.max()),
        "trunk_lean_mean_deg": float(lean.mean()),
        "standing_ratio": standing_ratio,
        "rise_rate": rise_rate,
    }


# 收案協定：蹲至最低點後停留約 3 秒。
# 治療師說明：若不要求停留時間，長者容易蹲下即刻起身、或反覆快速上下，
# 兩者皆有跌倒風險，因此協定明訂需在最低點維持。
SQUAT_HOLD_TARGET_S = 3.0

# 判定「仍在最低點」的膝角容許範圍（度）
SQUAT_HOLD_TOLERANCE_DEG = 12.0


# 低於此活動範圍視為「未實際完成下蹲」，此時停留時間不具意義
SQUAT_MIN_ROM_DEG = 25.0


def _hold_phase(knee: np.ndarray, tolerance: float = SQUAT_HOLD_TOLERANCE_DEG) -> Tuple[int, int]:
    """找出停留在最低點的連續區間，回傳 (起點, 終點) 索引。

    以最深處為中心，往前後延伸至膝角超出容許範圍為止。

    容許範圍會依實際活動範圍收窄：若受試者幾乎沒有下蹲（如 A003 的 ROM 僅 12 度），
    固定的 12 度容許範圍會把整段影片都判定為「停留」，產生假的長停留時間。
    """
    if len(knee) == 0:
        return 0, 0

    rom = float(np.ptp(knee))
    effective_tolerance = min(tolerance, max(rom * 0.2, 1.0))

    deepest = int(np.argmin(knee))
    threshold = knee[deepest] + effective_tolerance

    start = deepest
    while start > 0 and knee[start - 1] <= threshold:
        start -= 1

    end = deepest
    while end < len(knee) - 1 and knee[end + 1] <= threshold:
        end += 1

    return start, end


def squat_features(xyz: np.ndarray, vis: np.ndarray) -> Dict[str, float]:
    """半蹲：下肢肌力、關節活動度，以及最低點的維持能力。

    協定為「蹲至最低點停留約 3 秒」，因此**維持階段**是核心觀察對象——
    能否撐住、撐多久、撐的時候穩不穩，比單純的下蹲深度更貼近肌力表現。
    """
    duration = len(xyz) / FPS
    knee = _knee_angles(xyz, vis)
    lean = _trunk_lean(xyz)
    extent = _vertical_extent(xyz, vis)
    smoothed_knee = _smooth(knee)

    deepest_idx = int(np.argmin(smoothed_knee))
    depth = float(np.ptp(extent))

    rom = float(np.ptp(smoothed_knee))
    completed_squat = rom >= SQUAT_MIN_ROM_DEG

    start, end = _hold_phase(smoothed_knee)
    # 未實際完成下蹲者，其「停留時間」不具臨床意義，一律歸零並另以旗標標示
    hold_duration = (end - start + 1) / FPS if completed_squat else 0.0
    hold_slice = smoothed_knee[start : end + 1]

    # 維持期間的膝角抖動：肌力不足者較難維持穩定姿勢
    hold_tremor = float(hold_slice.std()) if len(hold_slice) > 1 else 0.0

    # 維持期間的鉛直高度漂移：撐不住會慢慢往下沉或提早站起
    hold_drift = float(np.ptp(extent[start : end + 1])) if end > start else 0.0

    # 下降與起身的角速度（度/秒）
    descent_speed = (
        float((smoothed_knee[0] - smoothed_knee[deepest_idx]) / max(deepest_idx / FPS, 1e-3))
        if deepest_idx > 0 else 0.0
    )
    ascent_frames = len(smoothed_knee) - 1 - deepest_idx
    ascent_speed = (
        float((smoothed_knee[-1] - smoothed_knee[deepest_idx]) / max(ascent_frames / FPS, 1e-3))
        if ascent_frames > 0 else 0.0
    )

    return {
        "duration_s": duration,
        "knee_min_deg": float(smoothed_knee.min()),
        "knee_rom_deg": float(np.ptp(smoothed_knee)),
        "squat_depth": depth,
        "trunk_lean_at_deepest_deg": float(lean[deepest_idx]),
        "trunk_lean_max_deg": float(lean.max()),
        "descent_ratio": deepest_idx / max(len(knee) - 1, 1),
        # 以下為協定的維持階段指標
        "completed_squat": 1.0 if completed_squat else 0.0,
        "hold_duration_s": hold_duration,
        "hold_ratio": hold_duration / SQUAT_HOLD_TARGET_S,
        "meets_hold_protocol": 1.0 if (completed_squat and hold_duration >= SQUAT_HOLD_TARGET_S) else 0.0,
        "hold_tremor_deg": hold_tremor,
        "hold_drift": hold_drift,
        "descent_speed_deg_s": descent_speed,
        "ascent_speed_deg_s": ascent_speed,
    }


def walk_features(xyz: np.ndarray, vis: np.ndarray) -> Dict[str, float]:
    """行走：步態節律與對稱性。

    注意：world landmarks 把髖部固定在原點，**絕對步速無法直接取得**。
    此處以「步頻 × 步長」估計，單位是軀幹長／秒（無量綱），
    僅供組間比較，不等同 AWGS 2019 的 m/s 步速門檻。
    """
    duration = len(xyz) / FPS
    knee = _knee_angles(xyz, vis)
    lean = _trunk_lean(xyz)

    # 前後方向（+Z）的雙踝分離量，隨每一步震盪
    ankle_sep = xyz[:, L_ANKLE, 2] - xyz[:, R_ANKLE, 2]
    steps, step_period, step_std = _count_cycles(ankle_sep)

    cadence = steps / duration if duration > 0 else 0.0
    step_length = float(np.ptp(_smooth(ankle_sep)))
    gait_speed_proxy = cadence * step_length

    # 左右對稱性：兩側膝角活動範圍的差異比例。
    # 只有在雙側都拍得到時才有意義，因此附上可見度供模型判斷可信度。
    l_knee = _angle_deg(xyz[:, L_HIP], xyz[:, L_KNEE], xyz[:, L_ANKLE])
    r_knee = _angle_deg(xyz[:, R_HIP], xyz[:, R_KNEE], xyz[:, R_ANKLE])
    l_rom, r_rom = float(np.ptp(l_knee)), float(np.ptp(r_knee))
    asymmetry = abs(l_rom - r_rom) / max(l_rom + r_rom, _EPS)

    return {
        "duration_s": duration,
        "steps": float(steps),
        "cadence_hz": cadence,
        "step_period_s": step_period,
        "step_period_std_s": step_std,
        "step_length": step_length,
        "gait_speed_proxy": gait_speed_proxy,
        "knee_rom_deg": float(np.ptp(knee)),
        "knee_asymmetry": asymmetry,
        "trunk_lean_mean_deg": float(lean.mean()),
    }


_ACTION_EXTRACTORS = {
    "Sit": sit_features,
    "Squat": squat_features,
    "Walk": walk_features,
}

# 特徵分組。用途見 DEVELOPMENT_LOG.md「年齡混淆」一節：
# 本資料集的陽性組平均年長 6 歲，年齡單獨即可達 AUC 0.80。
# 因此必須分組做消融，才能回答「動作資料本身有沒有貢獻」。
DEMOGRAPHIC_FEATURES: Tuple[str, ...] = (
    "age", "bmi", "height_cm", "weight_kg", "is_female",
)

# 可見度欄位與標籤相關（Walk 可見度最低的三筆皆為陽性），
# 若放進模型可能被當成捷徑學習。預設排除，僅保留供診斷報告使用。
QUALITY_FEATURES: Tuple[str, ...] = ("vis_mean", "vis_lower_body")

FEATURE_GROUPS = ("motion", "demographic", "combined")


def select_feature_group(
    matrix: np.ndarray,
    names: Sequence[str],
    group: str,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """依分組挑選特徵欄位。

    Args:
        group: ``"motion"``（純運動學）、``"demographic"``（純人口學）、
            ``"combined"``（兩者合併）。三者皆排除品質欄位。

    Returns:
        (篩選後的矩陣, 對應的特徵名稱)
    """
    if group not in FEATURE_GROUPS:
        raise ValueError(f"未知的特徵分組：{group}（可選 {FEATURE_GROUPS}）")

    keep = []
    for i, name in enumerate(names):
        if name in QUALITY_FEATURES:
            continue
        is_demo = name in DEMOGRAPHIC_FEATURES
        if group == "demographic" and not is_demo:
            continue
        if group == "motion" and is_demo:
            continue
        keep.append(i)

    return matrix[:, keep], tuple(names[i] for i in keep)


def extract_features(sample: Sample) -> Dict[str, float]:
    """對單筆樣本萃取完整特徵（動作專屬 + 基本資料 + 品質指標）。"""
    raw = load_landmarks(sample.json_path)
    normalized = normalize_sequence(raw)
    xyz = normalized[..., :3].astype(np.float64)
    vis = normalized[..., 3].astype(np.float64)

    features = _ACTION_EXTRACTORS[sample.action](xyz, vis)
    features.update(_static_features(sample))
    features.update(_quality_features(vis))
    return features


def feature_names(action: str) -> Tuple[str, ...]:
    """回傳某動作的特徵欄位順序（供建構特徵矩陣時對齊）。"""
    dummy_xyz = np.zeros((60, 33, 3))
    dummy_xyz[:, :, 1] = np.linspace(0, 1, 33)[None, :]
    dummy_vis = np.ones((60, 33))
    keys = list(_ACTION_EXTRACTORS[action](dummy_xyz, dummy_vis))
    keys += ["age", "bmi", "height_cm", "weight_kg", "is_female"]
    keys += ["vis_mean", "vis_lower_body"]
    return tuple(keys)


def build_feature_matrix(
    samples: Sequence[Sample],
    action: str,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """把某動作的所有樣本組成特徵矩陣。

    Returns:
        (X, y, 受試者ID, 特徵名稱)
    """
    action_samples = [s for s in samples if s.action == action]
    if not action_samples:
        raise ValueError(f"名冊中沒有 {action} 的樣本")

    names = feature_names(action)
    rows, labels, subjects = [], [], []
    for sample in action_samples:
        feats = extract_features(sample)
        rows.append([feats[name] for name in names])
        labels.append(sample.label)
        subjects.append(sample.subject_id)

    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        tuple(subjects),
        names,
    )

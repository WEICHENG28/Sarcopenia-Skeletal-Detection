"""骨架前處理：把 MediaPipe world landmarks JSON 轉成可訓練的乾淨張量。

解決的三個實際問題：

1. **鏡頭方向不一致**
   收案地點不同導致影片有 90 度旋轉、正負翻轉。實測 A004~A020 的頭在 X 軸負向、
   A021~A033 的頭在 X 軸正向，而且這個分組剛好和陽性率相關——不修正的話模型會
   偷學「座標軸朝哪邊」當作判斷依據（data leakage）。
   解法：用解剖結構（軀幹方向、髖部連線）建立每筆樣本自己的標準座標系，
   把整段序列旋轉到統一慣例：+Y 朝上、+X 朝受試者左側、+Z 朝前。

2. **身材差異**
   同樣的動作，高個子的座標數值本來就比較大。除以自己的軀幹長度做尺度正規化，
   讓「動作模式」可比較；身高體重本身仍以 static 特徵另外餵給模型。

3. **遮擋造成的假座標**
   側面拍攝時被身體擋住的關節，MediaPipe 仍會「猜」一個座標吐出來且外觀正常。
   visibility 保留成第 4 個通道，讓模型知道哪些點不可信。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

NUM_LANDMARKS = 33

# MediaPipe Pose 關節索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 身體節點（11~32）。捨棄臉部 0~10：
# 對肌少症無貢獻，且 MediaPipe 官方拓撲中臉部與身體不相連，
# 一併保留會讓圖分裂成三個互不相通的子圖。
BODY_NODES: Tuple[int, ...] = tuple(range(11, 33))

# 身體子圖的解剖連線（以原始 MediaPipe 索引表示）
BODY_EDGES_RAW: Tuple[Tuple[int, int], ...] = (
    (11, 12),  # 肩線
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (15, 17), (15, 19), (15, 21), (17, 19),  # 左手
    (16, 18), (16, 20), (16, 22), (18, 20),  # 右手
    (11, 23), (12, 24), (23, 24),  # 軀幹
    (23, 25), (25, 27),  # 左腿
    (24, 26), (26, 28),  # 右腿
    (27, 29), (29, 31), (27, 31),  # 左足
    (28, 30), (30, 32), (28, 32),  # 右足
)

_EPS = 1e-8


def load_landmarks(json_path: Path | str) -> np.ndarray:
    """讀取骨架 JSON，回傳 shape (T, 33, 4) 的陣列，通道為 (x, y, z, visibility)。"""
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"JSON 根層必須是非空的 frame list：{path}")

    frames = []
    for idx, frame in enumerate(payload):
        if not isinstance(frame, list) or len(frame) < NUM_LANDMARKS:
            raise ValueError(f"{path} 第 {idx} 幀關節數不足 {NUM_LANDMARKS}")
        frames.append(
            [
                [
                    float(kp.get("x", 0.0)),
                    float(kp.get("y", 0.0)),
                    float(kp.get("z", 0.0)),
                    float(kp.get("visibility", 1.0)),
                ]
                for kp in frame[:NUM_LANDMARKS]
            ]
        )

    return np.asarray(frames, dtype=np.float32)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        return vec
    return vec / norm


def _estimate_up(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """估計整段序列的「朝上」方向。

    優先用鼻子到腳踝中點（跨度最大、方向最穩），
    腳踝可見度太低時退回軀幹向量（肩中點減髖中點）。
    """
    ankle_vis = 0.5 * (vis[:, L_ANKLE] + vis[:, R_ANKLE])
    if float(np.median(ankle_vis)) >= 0.5:
        mid_ankle = 0.5 * (xyz[:, L_ANKLE] + xyz[:, R_ANKLE])
        candidate = np.median(xyz[:, NOSE] - mid_ankle, axis=0)
        if float(np.linalg.norm(candidate)) >= _EPS:
            return _unit(candidate)

    mid_shoulder = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])
    mid_hip = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
    return _unit(np.median(mid_shoulder - mid_hip, axis=0))


def _canonical_rotation(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """建立 3x3 旋轉矩陣，把序列轉到 +Y 朝上、+X 朝受試者左側、+Z 朝前。

    以解剖結構（而非攝影機）定義座標軸，因此天生不受鏡頭旋轉／翻轉影響，
    且左右方向由髖部連線決定，不會因為鏡像而把左右腿對調。
    """
    up = _estimate_up(xyz, vis)

    # 髖部連線（右髖指向左髖）作為側向參考，扣掉與 up 平行的分量後正交化
    lateral_raw = np.median(xyz[:, L_HIP] - xyz[:, R_HIP], axis=0)
    lateral = lateral_raw - float(np.dot(lateral_raw, up)) * up
    if float(np.linalg.norm(lateral)) < _EPS:
        # 髖部連線幾乎與 up 平行（理論上不該發生）：任取一個與 up 正交的方向
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, up))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = fallback - float(np.dot(fallback, up)) * up
    lateral = _unit(lateral)

    forward = _unit(np.cross(lateral, up))  # 右手系：X × Y = Z

    return np.stack([lateral, up, forward]).astype(np.float32)


def normalize_sequence(
    raw: np.ndarray,
    *,
    scale_normalize: bool = True,
) -> np.ndarray:
    """把原始 landmarks 正規化成標準座標系。

    Args:
        raw: (T, 33, 4)，來自 :func:`load_landmarks`。
        scale_normalize: 是否除以軀幹長度，讓不同身材的動作可比較。

    Returns:
        (T, 33, 4)，xyz 已旋轉／置中／縮放，visibility 原樣保留。
    """
    xyz = raw[..., :3].astype(np.float64)
    vis = raw[..., 3]

    rotation = _canonical_rotation(xyz, vis)
    xyz = xyz @ rotation.T.astype(np.float64)

    # 以髖部中點為原點（world landmarks 本來就接近如此，這裡明確化）
    mid_hip = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
    xyz = xyz - mid_hip[:, None, :]

    if scale_normalize:
        mid_shoulder = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])
        trunk_len = float(np.median(np.linalg.norm(mid_shoulder, axis=1)))
        if trunk_len > _EPS:
            xyz = xyz / trunk_len

    out = np.empty_like(raw)
    out[..., :3] = xyz.astype(np.float32)
    out[..., 3] = vis
    return out


def resample_time(seq: np.ndarray, target_frames: int) -> np.ndarray:
    """線性內插重採樣到固定長度。

    不用最近鄰抽格（原本 train.py 的做法），避免時間軸抖動與鋸齒。
    """
    t_in = seq.shape[0]
    if t_in == target_frames:
        return seq.astype(np.float32)
    if t_in == 1:
        return np.repeat(seq, target_frames, axis=0).astype(np.float32)

    src = np.linspace(0.0, t_in - 1, num=t_in)
    dst = np.linspace(0.0, t_in - 1, num=target_frames)
    flat = seq.reshape(t_in, -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst, src, flat[:, col])
    return out.reshape(target_frames, *seq.shape[1:])


def select_nodes(seq: np.ndarray, nodes: Sequence[int] = BODY_NODES) -> np.ndarray:
    """取出指定關節子集，例如捨棄臉部只留身體 22 點。"""
    return seq[:, list(nodes), :]


def to_model_tensor(
    seq: np.ndarray,
    *,
    include_visibility: bool = True,
    include_bone: bool = True,
    nodes: Sequence[int] = BODY_NODES,
) -> np.ndarray:
    """轉成模型輸入格式 [C, T, V]。

    通道組成：
      - 座標 x/y/z（3）
      - visibility（1，選用）——讓模型知道哪些關節是被遮擋後猜出來的
      - bone 向量（3，選用）——各關節相對其父關節的位移，借用 2s-AGCN 的
        雙流概念，對絕對位置偏移較不敏感，通常比純座標更有辨識力
    """
    selected = select_nodes(seq, nodes)
    xyz = selected[..., :3]
    channels = [xyz]

    if include_bone:
        index_map = {node: i for i, node in enumerate(nodes)}
        bone = np.zeros_like(xyz)
        for child, parent in BODY_EDGES_RAW:
            if child in index_map and parent in index_map:
                ci, pi = index_map[child], index_map[parent]
                bone[:, ci] = xyz[:, ci] - xyz[:, pi]
        channels.append(bone)

    if include_visibility:
        channels.append(selected[..., 3:4])

    stacked = np.concatenate(channels, axis=-1)  # (T, V, C)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)  # (C, T, V)


def build_body_adjacency(nodes: Sequence[int] = BODY_NODES) -> np.ndarray:
    """建立身體子圖的鄰接矩陣，採 ST-GCN 慣用的對稱正規化 D^-1/2 A D^-1/2。"""
    index_map = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adj = np.eye(n, dtype=np.float32)
    for a, b in BODY_EDGES_RAW:
        if a in index_map and b in index_map:
            i, j = index_map[a], index_map[b]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    deg = adj.sum(axis=1)
    d_inv_sqrt = np.power(np.maximum(deg, _EPS), -0.5)
    return (adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


def load_and_prepare(
    json_path: Path | str,
    *,
    target_frames: int,
    include_visibility: bool = True,
    include_bone: bool = True,
    nodes: Sequence[int] = BODY_NODES,
) -> np.ndarray:
    """一次完成：讀檔 → 正規化 → 重採樣 → 轉模型張量 [C, T, V]。"""
    raw = load_landmarks(json_path)
    normalized = normalize_sequence(raw)
    resampled = resample_time(normalized, target_frames)
    return to_model_tensor(
        resampled,
        include_visibility=include_visibility,
        include_bone=include_bone,
        nodes=nodes,
    )

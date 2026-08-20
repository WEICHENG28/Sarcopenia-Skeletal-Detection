"""資料理解類圖表：骨架點位、動作姿勢、族群分布、資料品質。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.dataset_manifest import ACTIONS, FPS
from src.figures import data_access as da
from src.figures.style import (
    ACTION_COLORS, COLOR_ACCENT, COLOR_MUTED, COLOR_NEGATIVE, COLOR_NEUTRAL,
    COLOR_POSITIVE, COLOR_WARN, LABEL_COLORS, LABEL_MARKERS, LABEL_NAMES,
    annotate, save,
)
from src.skeleton_io import BODY_EDGES_RAW, BODY_NODES

# MediaPipe Pose 33 點的官方連線（含臉部，用於呈現完整拓撲）
FACE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10))

LANDMARK_NAMES = {
    0: "鼻", 1: "左眼內", 2: "左眼", 3: "左眼外", 4: "右眼內", 5: "右眼",
    6: "右眼外", 7: "左耳", 8: "右耳", 9: "嘴左", 10: "嘴右",
    11: "左肩", 12: "右肩", 13: "左肘", 14: "右肘", 15: "左腕", 16: "右腕",
    17: "左小指", 18: "右小指", 19: "左食指", 20: "右食指",
    21: "左拇指", 22: "右拇指", 23: "左髖", 24: "右髖",
    25: "左膝", 26: "右膝", 27: "左踝", 28: "右踝",
    29: "左腳跟", 30: "右腳跟", 31: "左腳尖", 32: "右腳尖",
}

# 標註時只顯示臨床上關鍵的關節，避免圖面過於擁擠
KEY_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32)


def _reference_pose() -> np.ndarray:
    """挑出全體資料中最接近標準站姿的一幀，作為點位示意。

    評分準則：軀幹鉛直、雙腿伸直、左右對稱、關節可見度高。
    """
    def knee_angle(xyz: np.ndarray, hip: int, knee: int, ankle: int) -> np.ndarray:
        upper = xyz[:, hip] - xyz[:, knee]
        lower = xyz[:, ankle] - xyz[:, knee]
        cos = np.sum(upper * lower, axis=1) / (
            np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1) + 1e-9
        )
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    best_score, best_pose = -np.inf, None

    for sample in da.manifest():
        sequence = da.landmarks(sample.subject_id, sample.action)
        xyz = sequence[..., :3]
        visibility = sequence[..., 3]

        shoulder = 0.5 * (xyz[:, 11] + xyz[:, 12])
        hip = 0.5 * (xyz[:, 23] + xyz[:, 24])
        trunk = shoulder - hip

        # 各項皆為無量綱比值，避免受個別樣本的尺度異常影響
        uprightness = trunk[:, 1] / (np.linalg.norm(trunk, axis=1) + 1e-9)
        straightness = (
            knee_angle(xyz, 23, 25, 27) + knee_angle(xyz, 24, 26, 28)
        ) / 360.0
        symmetry = 1.0 - np.clip(
            np.abs(xyz[:, 27, 1] - xyz[:, 28, 1]) / 0.3, 0, 1
        )
        quality = visibility.mean(axis=1)

        score = uprightness + straightness * 2.0 + symmetry + quality
        frame = int(np.argmax(score))
        if score[frame] > best_score:
            best_score, best_pose = float(score[frame]), xyz[frame]

    return best_pose


def skeleton_layout() -> None:
    """圖 1：MediaPipe Pose 33 點位置與本研究採用的 22 點。"""
    pose = _reference_pose()
    # 依實際姿勢決定畫布範圍，兩側各留出標籤欄位的寬度
    span_y = pose[:, 1].max() - pose[:, 1].min()
    half_width = span_y * 0.62
    y_low, y_high = pose[:, 1].min() - span_y * 0.10, pose[:, 1].max() + span_y * 0.06
    label_x = half_width * 0.94

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6))

    for ax, (title, highlight) in zip(axes, [
        ("MediaPipe Pose 全部 33 個關鍵點", False),
        ("本研究採用的身體 22 點（編號 11–32）", True),
    ]):
        # 連線
        for a, b in FACE_EDGES:
            color = COLOR_MUTED if highlight else COLOR_ACCENT
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]],
                    color=color, lw=1.6, zorder=1,
                    alpha=0.3 if highlight else 0.85)
        for a, b in BODY_EDGES_RAW:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]],
                    color=COLOR_ACCENT, lw=2.4, zorder=1, alpha=0.9)

        # 點
        for idx in range(33):
            used = idx in BODY_NODES
            if highlight:
                color = COLOR_POSITIVE if used else COLOR_MUTED
                size = 78 if used else 34
            else:
                color = COLOR_ACCENT
                size = 62
            ax.scatter(pose[idx, 0], pose[idx, 1], s=size, color=color,
                       edgecolor="white", linewidth=1.1, zorder=3)

        # 編號與名稱：以引線拉到兩側固定欄位，避免標籤互相重疊
        labelled = [i for i in KEY_LANDMARKS
                    if not (highlight and i not in BODY_NODES)]
        for side, sign in (("left", -1), ("right", 1)):
            column = [i for i in labelled if np.sign(pose[i, 0] or sign) == sign]
            column.sort(key=lambda i: -pose[i, 1])
            if not column:
                continue
            slots = np.linspace(
                pose[:, 1].max() * 0.98, pose[:, 1].min() * 0.98, len(column)
            )
            for idx, y_slot in zip(column, slots):
                x_text = sign * label_x
                ax.annotate(
                    f"{idx} {LANDMARK_NAMES[idx]}",
                    xy=(pose[idx, 0], pose[idx, 1]),
                    xytext=(x_text, y_slot),
                    ha="left" if sign > 0 else "right", va="center",
                    fontsize=7.8, color=COLOR_NEUTRAL, zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#C9D2DA",
                                    lw=0.7, shrinkA=0, shrinkB=3),
                )

        ax.set_title(title, pad=14)
        ax.set_aspect("equal")
        ax.set_xlim(-half_width * 1.55, half_width * 1.55)
        ax.set_ylim(y_low, y_high)
        ax.axis("off")

    axes[0].text(0, y_low + span_y * 0.02, "正面視角：畫面右側為受試者的左半身",
                 ha="center", fontsize=8.8, color=COLOR_NEUTRAL, style="italic")
    axes[1].legend(handles=[
        Patch(facecolor=COLOR_POSITIVE, label="採用（身體 22 點）"),
        Patch(facecolor=COLOR_MUTED, label="捨棄（臉部 11 點）"),
    ], loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "骨架關鍵點定義　—　捨棄臉部是因為該子圖與身體不相連，圖卷積無法傳遞訊息",
        fontsize=12.5, y=1.0,
    )
    save(fig, "01_骨架點位定義", "33 點全貌與採用的 22 點")


def action_poses() -> None:
    """圖 2：三個動作的代表性姿勢。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6))
    subject = "A028"

    picks = {
        "Sit": ("起立坐下（五次）", "最低點"),
        "Walk": ("向前行走", "跨步中"),
        "Squat": ("半蹲（停留 3 秒）", "最低點"),
    }

    for ax, action in zip(axes, ACTIONS):
        sequence = da.landmarks(subject, action)
        xyz = sequence[..., :3]
        hip = 0.5 * (xyz[:, 23] + xyz[:, 24])
        ankle = 0.5 * (xyz[:, 27] + xyz[:, 28])
        extent = np.linalg.norm(hip - ankle, axis=1)
        frame = int(np.argmin(extent)) if action != "Walk" else int(len(xyz) // 2)
        pose = xyz[frame]

        for a, b in BODY_EDGES_RAW:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]],
                    color=ACTION_COLORS[action], lw=2.8, alpha=0.9, zorder=1)
        ax.scatter(pose[list(BODY_NODES), 0], pose[list(BODY_NODES), 1],
                   s=42, color=ACTION_COLORS[action],
                   edgecolor="white", linewidth=1.0, zorder=2)

        title, phase = picks[action]
        ax.set_title(f"{title}\n{phase}（第 {frame} 幀 / 共 {len(xyz)} 幀）",
                     fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 0.9)
        ax.axis("off")

    fig.suptitle(
        f"三項檢測動作的骨架姿勢（受試者 {subject}）　—　"
        "皆為 AWGS 2019 體能表現的正式測量項目",
        fontsize=12.5, y=1.0,
    )
    save(fig, "02_三動作姿勢", "Sit / Walk / Squat 代表幀")


def age_confound() -> None:
    """圖 3：年齡混淆——本研究最大的干擾因子。"""
    profiles = da.subjects()
    labels = da.subject_labels()
    ages = {
        0: np.array([profiles[s].age for s in labels if labels[s] == 0]),
        1: np.array([profiles[s].age for s in labels if labels[s] == 1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                            gridspec_kw={"width_ratios": [1.25, 1]})

    # 左：年齡分層堆疊
    ax = axes[0]
    bands = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 92)]
    centers = np.arange(len(bands))
    negative = [int(((ages[0] >= lo) & (ages[0] < hi)).sum()) for lo, hi in bands]
    positive = [int(((ages[1] >= lo) & (ages[1] < hi)).sum()) for lo, hi in bands]

    ax.bar(centers, negative, 0.62, label=LABEL_NAMES[0], color=COLOR_NEGATIVE)
    ax.bar(centers, positive, 0.62, bottom=negative,
           label=LABEL_NAMES[1], color=COLOR_POSITIVE)
    for i, (n, p) in enumerate(zip(negative, positive)):
        if p:
            ax.text(i, n + p / 2, str(p), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        if n:
            ax.text(i, n / 2, str(n), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in bands])
    ax.set_xlabel("年齡（歲）")
    ax.set_ylabel("人數")
    ax.set_title("80 歲以上 8 人中有 7 人為陽性")
    ax.legend()

    # 右：分布比較
    ax = axes[1]
    for label in (0, 1):
        parts = ax.violinplot([ages[label]], positions=[label],
                              widths=0.68, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(LABEL_COLORS[label])
            body.set_alpha(0.32)
        jitter = np.random.default_rng(0).normal(0, 0.045, len(ages[label]))
        ax.scatter(label + jitter, ages[label], s=42,
                   color=LABEL_COLORS[label], marker=LABEL_MARKERS[label],
                   edgecolor="white", linewidth=0.9, zorder=3)
        ax.hlines(ages[label].mean(), label - 0.28, label + 0.28,
                  color=LABEL_COLORS[label], lw=2.6, zorder=4)
        ax.text(label, ages[label].mean() + 1.1,
                f"{ages[label].mean():.1f} 歲", ha="center",
                fontsize=9.5, fontweight="bold", color=LABEL_COLORS[label])

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{LABEL_NAMES[0]}\n(n={len(ages[0])})",
                        f"{LABEL_NAMES[1]}\n(n={len(ages[1])})"])
    ax.set_ylabel("年齡（歲）")
    ax.set_title("兩組平均相差 13 歲")
    annotate(ax, "單看年齡即可達 AUC 0.879\n因此必須做特徵分組消融", "upper left")

    fig.suptitle(
        "年齡是本研究最大的混淆因子　—　"
        "若不控制，無法區分模型是在看動作還是在看年齡",
        fontsize=12.5, y=1.02,
    )
    save(fig, "03_年齡混淆", "陽性組平均年長 13 歲")


def data_quality() -> None:
    """圖 4：各關節可見度——低可見度代表座標為系統推測值。"""
    joint_groups = {
        "軀幹": (11, 12, 23, 24),
        "上肢": (13, 14, 15, 16),
        "膝": (25, 26),
        "踝": (27, 28),
        "足": (29, 30, 31, 32),
    }

    matrix = np.zeros((len(ACTIONS), len(joint_groups)))
    for i, action in enumerate(ACTIONS):
        for j, (_, joints) in enumerate(joint_groups.items()):
            values = []
            for sample in da.manifest():
                if sample.action != action:
                    continue
                sequence = da.landmarks(sample.subject_id, action)
                values.append(sequence[:, list(joints), 3].mean())
            matrix[i, j] = float(np.mean(values))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                            gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0.75, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(joint_groups)))
    ax.set_xticklabels(joint_groups.keys())
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS)
    for i in range(len(ACTIONS)):
        for j in range(len(joint_groups)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if matrix[i, j] < 0.88 else "#333333")
    ax.set_title("各動作 × 關節群的平均可見度")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="可見度")

    # 右：逐樣本下肢可見度
    ax = axes[1]
    lower = (23, 24, 25, 26, 27, 28)
    for i, action in enumerate(ACTIONS):
        values, colors = [], []
        for sample in da.manifest():
            if sample.action != action:
                continue
            sequence = da.landmarks(sample.subject_id, action)
            values.append(sequence[:, list(lower), 3].mean())
            colors.append(LABEL_COLORS[sample.label])
        jitter = np.random.default_rng(1).normal(0, 0.07, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values,
                   s=38, c=colors, alpha=0.82,
                   edgecolor="white", linewidth=0.7)

    ax.axhline(0.85, color=COLOR_WARN, ls="--", lw=1.4)
    ax.text(2.42, 0.855, "0.85 判讀門檻", fontsize=8.4,
            color=COLOR_WARN, ha="right")
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("下肢平均可見度")
    ax.set_title("行走是唯一需改善拍攝角度的動作")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=LABEL_COLORS[k],
               label=LABEL_NAMES[k], markersize=7) for k in (0, 1)
    ], loc="lower left")
    annotate(ax, "可見度低 = 關節被遮擋\nMediaPipe 仍會輸出推測座標", "lower right")

    fig.suptitle(
        "錄影品質稽核　—　關節被遮擋時系統仍會輸出看似正常的推測座標",
        fontsize=12.5, y=1.02,
    )
    save(fig, "04_資料品質", "行走的下肢可見度偏低")


def sequence_lengths() -> None:
    """圖 5：各動作的長度分布——說明為何三動作分開設定序列長度。"""
    from src.datasets import TARGET_FRAMES

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    for i, action in enumerate(ACTIONS):
        durations, colors = [], []
        for sample in da.manifest():
            if sample.action != action:
                continue
            sequence = da.landmarks(sample.subject_id, action)
            durations.append(len(sequence) / FPS)
            colors.append(LABEL_COLORS[sample.label])
        jitter = np.random.default_rng(2).normal(0, 0.075, len(durations))
        ax.scatter(np.array(durations), np.full(len(durations), i) + jitter,
                   s=44, c=colors, alpha=0.82,
                   edgecolor="white", linewidth=0.7, zorder=3)

        median = float(np.median(durations))
        ax.plot([median, median], [i - 0.3, i + 0.3],
                color=COLOR_NEUTRAL, lw=2.4, zorder=4)
        ax.text(median, i + 0.36, f"中位數 {median:.1f}s",
                ha="center", fontsize=8.6, color=COLOR_NEUTRAL)

        target = TARGET_FRAMES[action] / FPS
        ax.scatter([target], [i], marker="|", s=460,
                   color=ACTION_COLORS[action], linewidth=2.6, zorder=5)

    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS)
    ax.set_xlabel("影片長度（秒）")
    ax.set_title("三個動作的長度差異達 3 倍，因此各自設定模型輸入長度")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=LABEL_COLORS[k],
               label=LABEL_NAMES[k], markersize=7) for k in (0, 1)
    ] + [
        Line2D([], [], marker="|", ls="", color=COLOR_NEUTRAL,
               label="模型輸入長度", markersize=13, markeredgewidth=2.4)
    ], loc="upper right")
    annotate(ax, "舊版三動作共用 3.2 秒，會把起立坐下大幅截斷", "lower right")

    save(fig, "05_序列長度分布", "Sit 中位數 14s、Walk 5s、Squat 6s")


def generate_all() -> None:
    print("【資料理解】")
    skeleton_layout()
    action_poses()
    age_confound()
    data_quality()
    sequence_lengths()

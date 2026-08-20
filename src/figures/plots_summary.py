"""總結類圖表：系統流程、信賴區間比較、半蹲維持期的發現。"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.dataset_manifest import ACTIONS
from src.figures import data_access as da
from src.figures.style import (
    ACTION_COLORS, COLOR_ACCENT, COLOR_MUTED, COLOR_NEGATIVE, COLOR_NEUTRAL,
    COLOR_POSITIVE, COLOR_WARN, LABEL_COLORS, LABEL_NAMES, annotate, save,
)


def pipeline() -> None:
    """圖 18：系統整體流程——供醫療人員理解資料如何被處理。"""
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    stages = [
        (2, "現場錄影", "治療師以手機拍攝\n三項動作\n720p / 30fps", COLOR_NEUTRAL),
        (18, "骨架擷取", "MediaPipe Pose\n每幀 33 個關節點\n（含可見度分數）", ACTION_COLORS["Sit"]),
        (34, "前處理", "座標軸統一\n身材尺度正規化\n取身體 22 點", ACTION_COLORS["Walk"]),
        (50, "深度模型", "ST-GCN\n抽取時空特徵\n（每動作 16 維）", ACTION_COLORS["Squat"]),
        (66, "整合判斷", "三動作加權\n+ 年齡等基本資料\nLogistic Regression", COLOR_ACCENT),
        (82, "風險報告", "風險機率\n風險分級\n關注部位\n品質警告", COLOR_POSITIVE),
    ]

    for x, title, detail, color in stages:
        ax.add_patch(FancyBboxPatch(
            (x, 14), 14, 14, boxstyle="round,pad=0.3",
            facecolor=color, edgecolor="none", zorder=2,
        ))
        ax.text(x + 7, 25.4, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="white", zorder=3)
        ax.text(x + 7, 19.4, detail, ha="center", va="center",
                fontsize=8.4, color="white", zorder=3, linespacing=1.55)
        if x < 82:
            ax.add_patch(FancyArrowPatch(
                (x + 14.3, 21), (x + 15.7, 21), arrowstyle="-|>",
                mutation_scale=15, color=COLOR_NEUTRAL, lw=1.8, zorder=1,
            ))

    # 下方補充：關鍵設計決定
    notes = [
        (18, "使用 world landmarks\n不受拍攝距離影響"),
        (34, "解決不同場地的\n鏡頭方向不一致"),
        (50, "先以合成資料預訓練\n再用真實個案微調"),
        (66, "僅完成 2/3 動作\n仍可評估"),
    ]
    for x, text in notes:
        ax.plot([x + 7, x + 7], [13.4, 10], color="#D5DCE3", lw=1.0, zorder=1)
        ax.text(x + 7, 9, text, ha="center", va="top", fontsize=7.8,
                color=COLOR_NEUTRAL, style="italic", linespacing=1.5)

    ax.text(50, 34.5, "整體流程", ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 31.2,
            "從一支手機影片，到一份可解釋的肌少症風險報告",
            ha="center", fontsize=10, color=COLOR_NEUTRAL)

    save(fig, "18_系統流程", "六個階段總覽")


def confidence_intervals() -> None:
    """圖 19：信賴區間比較——說明為何不能宣稱模型優於年齡。"""
    entries = [
        ("系統 C：Shift-GCN 純骨架", 0.439, 0.22, 0.69, COLOR_MUTED),
        ("系統 A：純動作特徵", 0.671, 0.45, 0.87, COLOR_MUTED),
        ("系統 B：ST-GCN 純骨架", 0.789, 0.55, 0.97, COLOR_ACCENT),
        ("基準線：只看年齡等", 0.794, 0.59, 0.95, COLOR_NEUTRAL),
        ("系統 B：+ 人口學", 0.823, 0.64, 0.95, COLOR_POSITIVE),
        ("系統 A：+ 人口學", 0.825, 0.66, 0.95, COLOR_NEGATIVE),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    positions = np.arange(len(entries))

    for pos, (name, auc, low, high, color) in zip(positions, entries):
        ax.plot([low, high], [pos, pos], color=color, lw=2.6, zorder=2)
        ax.plot([low, low], [pos - 0.14, pos + 0.14], color=color, lw=2.2)
        ax.plot([high, high], [pos - 0.14, pos + 0.14], color=color, lw=2.2)
        ax.scatter([auc], [pos], s=130, color=color, zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.text(high + 0.012, pos, f"{auc:.3f}", va="center",
                fontsize=9.2, fontweight="bold", color=color)

    baseline = 0.794
    ax.axvline(baseline, color=COLOR_NEUTRAL, ls="--", lw=1.4, zorder=1)
    ax.axvspan(0.59, 0.95, color=COLOR_NEUTRAL, alpha=0.06, zorder=0)
    ax.axvline(0.5, color="#C8D0D8", lw=1.2, zorder=1)
    ax.text(0.505, 2.6, "隨機猜測", fontsize=8.2, color="#A6B0BA",
            rotation=90, va="center")

    ax.set_yticks(positions)
    ax.set_yticklabels([e[0] for e in entries], fontsize=9.5)
    ax.set_xlabel("AUC（點為估計值，橫線為 bootstrap 95% 信賴區間）")
    ax.set_xlim(0.15, 1.06)
    ax.set_title("所有系統的信賴區間高度重疊　—　31 人下無法宣稱誰較優", pad=12)
    annotate(ax,
             "灰底為基準線的信賴區間 [0.59, 0.95]\n"
             "所有系統的估計值都落在其中",
             "lower right")

    save(fig, "19_信賴區間比較", "誠實呈現統計不確定性")


def squat_discovery() -> None:
    """圖 20：半蹲維持期特徵的發現——臨床協定決定該量什麼。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：加入維持期特徵前後
    ax = axes[0]
    groups = ["純動作特徵", "+ 人口學"]
    before = [0.465, 0.671]
    after = [0.561, 0.737]
    x = np.arange(len(groups))

    ax.bar(x - 0.19, before, 0.36, label="僅測「蹲多深」", color=COLOR_MUTED)
    ax.bar(x + 0.19, after, 0.36, label="加測「撐多久」", color=COLOR_ACCENT)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=9,
                fontweight="bold", color=COLOR_ACCENT)
        ax.annotate(f"+{a - b:.3f}", xy=(i, max(a, b) + 0.055),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color=COLOR_POSITIVE)

    ax.axhline(0.5, color="#C8D0D8", lw=1.2, ls="--")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Squat 的 AUC")
    ax.set_ylim(0, 0.9)
    ax.set_title("補上協定的核心指標後，半蹲從無用變成有用")
    ax.legend(loc="upper left", fontsize=8.8)

    # 右：停留時間分布
    ax = axes[1]
    from src.features_handcrafted import build_feature_matrix

    matrix, labels, ids, names = build_feature_matrix(list(da.manifest()), "Squat")
    holds = matrix[:, names.index("hold_duration_s")]
    completed = matrix[:, names.index("completed_squat")] == 1.0

    bands = [(0, 1), (1, 2), (2, 3), (3, 8)]
    band_labels = ["< 1 秒", "1–2 秒", "2–3 秒", "3 秒以上\n（達標）"]
    centers = np.arange(len(bands))
    negative, positive = [], []
    for low, high in bands:
        mask = completed & (holds >= low) & (holds < high)
        positive.append(int(labels[mask].sum()))
        negative.append(int((1 - labels[mask]).sum()))

    ax.bar(centers, negative, 0.6, label=LABEL_NAMES[0], color=COLOR_NEGATIVE)
    ax.bar(centers, positive, 0.6, bottom=negative,
           label=LABEL_NAMES[1], color=COLOR_POSITIVE)
    for i, (n, p) in enumerate(zip(negative, positive)):
        if n:
            ax.text(i, n / 2, str(n), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        if p:
            ax.text(i, n + p / 2, str(p), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")

    ax.set_xticks(centers, band_labels)
    ax.set_xlabel("在最低點停留的時間")
    ax.set_ylabel("人數")
    ax.set_title("僅 8/30 人達到協定要求的 3 秒")
    ax.legend(loc="upper left", fontsize=8.8)
    annotate(ax,
             "撐不到 2 秒者以肌少症居多（13 人中 8 位）\n"
             "「撐不住」本身即為訊號，非收案失誤",
             "upper right")

    fig.suptitle(
        "半蹲：臨床協定決定了該測量什麼　—　"
        "協定要求「蹲到最低點停留 3 秒」，但初版只測了下蹲深度",
        fontsize=12.5, y=1.03,
    )
    fig.tight_layout()
    save(fig, "20_半蹲維持期發現", "量對東西比換模型重要")


def system_comparison() -> None:
    """圖 21：三系統總覽——一張圖看完所有結果。"""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))

    systems = ["系統 A\n手工特徵", "系統 B\nST-GCN", "系統 C\nShift-GCN"]
    motion = [0.671, 0.789, 0.439]
    combined = [0.825, 0.823, 0.828]
    x = np.arange(len(systems))

    ax.bar(x - 0.19, motion, 0.36, label="純動作／純骨架", color=COLOR_ACCENT)
    ax.bar(x + 0.19, combined, 0.36, label="加上年齡等人口學", color=COLOR_POSITIVE)

    for i, (m, c) in enumerate(zip(motion, combined)):
        ax.text(i - 0.19, m + 0.014, f"{m:.3f}", ha="center",
                fontsize=9.5, fontweight="bold", color=COLOR_ACCENT)
        ax.text(i + 0.19, c + 0.014, f"{c:.3f}", ha="center",
                fontsize=9.5, fontweight="bold", color=COLOR_POSITIVE)

    ax.axhline(0.794, color=COLOR_NEUTRAL, ls="--", lw=1.6)
    ax.text(2.48, 0.806, "基準線 0.794\n（只看年齡等）", fontsize=8.6,
            color=COLOR_NEUTRAL, ha="right")
    ax.axhline(0.5, color="#C8D0D8", lw=1.2, ls=":")

    ax.annotate(
        "系統 C 的 0.828 幾乎全來自年齡\n（純骨架僅 0.439，低於隨機）",
        xy=(2.19, 0.828), xytext=(1.62, 0.30),
        fontsize=8.6, color=COLOR_WARN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_WARN, lw=1.3),
    )

    ax.set_xticks(x, systems)
    ax.set_ylabel("Late Fusion AUC（out-of-fold）")
    ax.set_ylim(0, 1.0)
    ax.set_title("三系統比較　—　判斷模型是否真有貢獻，必須看「純骨架」欄", pad=12)
    ax.legend(loc="upper left")

    save(fig, "21_三系統比較", "只看合併欄會得出錯誤結論")


def generate_all() -> None:
    print("【總結】")
    pipeline()
    confidence_intervals()
    squat_discovery()
    system_comparison()

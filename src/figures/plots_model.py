"""模型類圖表：架構示意、ROC / PR 曲線、混淆矩陣、消融與逐人預測。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from src.dataset_manifest import ACTIONS
from src.figures import data_access as da
from src.figures.style import (
    ACTION_COLORS, COLOR_ACCENT, COLOR_MUTED, COLOR_NEGATIVE, COLOR_NEUTRAL,
    COLOR_POSITIVE, COLOR_WARN, LABEL_COLORS, LABEL_MARKERS, LABEL_NAMES,
    annotate, save,
)


# ---------------------------------------------------------------- 架構圖

def _box(ax, x, y, w, h, text, color, fontsize=8.5, text_color="white",
         alpha=1.0, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=3, linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, color=COLOR_NEUTRAL, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        color=color, lw=lw, zorder=1,
    ))


def architecture() -> None:
    """圖 11：分類器架構——供非本科系人員理解各層在做什麼。"""
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    # ── 階段一：單一動作 ──────────────────────────────────
    ax.text(2, 53, "階段一　每個動作各自判斷（三個動作各一個模型）",
            fontsize=12, fontweight="bold", color=COLOR_ACCENT)

    y = 34
    _box(ax, 2, y, 11, 13,
         "輸入\n骨架序列\n\n22 關節\n× 128 幀\n× 7 通道",
         COLOR_NEUTRAL, 8.2)
    ax.text(7.5, y - 2.4, "把影片變成\n數字矩陣", ha="center", va="top",
            fontsize=7.6, color=COLOR_NEUTRAL, style="italic")

    _arrow(ax, 13.2, y + 6.5, 16.8, y + 6.5)

    _box(ax, 17, y, 12, 13,
         "圖卷積 ①\n8 通道\n\n看「相鄰關節\n如何連動」",
         ACTION_COLORS["Sit"], 8.2)
    _arrow(ax, 29.2, y + 6.5, 32.8, y + 6.5)

    _box(ax, 33, y, 12, 13,
         "圖卷積 ②\n16 通道\n\n看「更大範圍\n的動作模式」",
         ACTION_COLORS["Sit"], 8.2)
    _arrow(ax, 45.2, y + 6.5, 48.8, y + 6.5)

    _box(ax, 49, y + 2, 11, 9,
         "平均池化\n\n壓縮成\n16 個數字",
         COLOR_MUTED, 8.2, text_color="#2E3A45")
    ax.text(54.5, y + 13.2, "「時空特徵向量」", ha="center", va="bottom",
            fontsize=8.4, color=COLOR_NEUTRAL, fontweight="bold")

    _arrow(ax, 60.2, y + 6.5, 63.8, y + 6.5)

    _box(ax, 64, y + 3.5, 10, 6, "Linear\n16 → 1", COLOR_ACCENT, 8.5)
    ax.text(69, y + 2.4, "加權相加成一個分數", ha="center", va="top",
            fontsize=7.6, color=COLOR_NEUTRAL, style="italic")

    _arrow(ax, 74.2, y + 6.5, 77.8, y + 6.5)

    _box(ax, 78, y + 3.5, 10, 6, "Sigmoid", COLOR_ACCENT, 8.5)
    ax.text(83, y + 2.4, "把分數壓成 0~1 的機率", ha="center", va="top",
            fontsize=7.6, color=COLOR_NEUTRAL, style="italic")

    _arrow(ax, 88.2, y + 6.5, 91.5, y + 6.5)
    _box(ax, 91.5, y + 3.5, 7.5, 6, "機率\n0~1", COLOR_POSITIVE, 8.5)

    # 三個動作各有一套相同結構的模型，以錯位外框示意
    for depth, action in zip((2.4, 1.2, 0.0), ("Squat", "Walk", "Sit")):
        ax.add_patch(FancyBboxPatch(
            (1.2 + depth, y - 1.2 + depth), 98.0 - depth, 15.4,
            boxstyle="round,pad=0.15",
            facecolor="none", edgecolor=ACTION_COLORS[action],
            lw=1.2, alpha=0.5, zorder=0,
        ))
        ax.text(2.2 + depth, y + 14.6 + depth, action, fontsize=8.2,
                color=ACTION_COLORS[action], fontweight="bold",
                ha="left", va="center")

    # ── 階段二：整合 ──────────────────────────────────────
    ax.text(2, 26, "階段二　整合三個動作與基本資料",
            fontsize=12, fontweight="bold", color=COLOR_ACCENT)

    y2 = 10
    for i, action in enumerate(ACTIONS):
        _box(ax, 3, y2 + 11 - i * 4.2, 13, 3.4,
             f"{action} 機率", ACTION_COLORS[action], 8.2)

    _arrow(ax, 16.2, y2 + 8, 21.8, y2 + 8)

    _box(ax, 22, y2 + 4.5, 15, 8,
         "Late Fusion\n加權平均\n\nSit 0.4 / Walk 0.4\n/ Squat 0.2",
         COLOR_NEUTRAL, 8.2)
    ax.text(29.5, y2 + 13.6, "只做兩項時分母自動調整", ha="center", va="bottom",
            fontsize=7.8, color=COLOR_WARN, style="italic")

    _box(ax, 3, y2 - 2.5, 13, 3.4, "年齡 · BMI · 身高\n體重 · 性別",
         COLOR_MUTED, 7.8, text_color="#2E3A45")

    _arrow(ax, 37.2, y2 + 8, 43.8, y2 + 8)
    _arrow(ax, 16.2, y2 - 0.8, 43.8, y2 + 6.0)

    _box(ax, 44, y2 + 3, 16, 10,
         "Stacking\nLogistic Regression\n\n輸入 8 個數字：\n3 個動作機率\n+ 5 項基本資料",
         COLOR_ACCENT, 8.2)
    ax.text(52, y2 + 1.9,
            "用簡單模型處理年齡，\n避免被高維骨架特徵淹沒",
            ha="center", va="top", fontsize=7.6,
            color=COLOR_NEUTRAL, style="italic")

    _arrow(ax, 60.2, y2 + 8, 66.8, y2 + 8)

    _box(ax, 67, y2 + 4, 14, 8,
         "最終風險機率\n0 ~ 1", COLOR_POSITIVE, 9.5)

    _arrow(ax, 81.2, y2 + 8, 85.8, y2 + 8)
    _box(ax, 86, y2 + 2.5, 13, 11,
         "風險分級\n\n0.70 以上　高\n0.513 以上　中高\n0.35 以上　中低\n0.35 以下　低",
         COLOR_NEUTRAL, 8.0)

    fig.suptitle(
        "分類器架構　—　從骨架序列到風險分級",
        fontsize=14, y=0.99,
    )
    save(fig, "11_分類器架構", "供非本科系人員理解")


# ---------------------------------------------------------------- 曲線

def _curve_sources() -> List[Tuple[str, np.ndarray, np.ndarray, str, dict]]:
    """整理各系統的 out-of-fold 預測，供 ROC / PR 使用。"""
    sources = []

    per_action, labels = da.oof_probabilities("system_stgcn_pretrained.json", "motion")
    ids, probs, target = da.aligned(da.fuse(per_action), labels)
    sources.append(("系統 B：ST-GCN 純骨架", probs, target, COLOR_ACCENT,
                    dict(lw=2.6)))

    _, stack_probs, stack_target = da.stacking_oof("system_stgcn_pretrained.json")
    sources.append(("系統 B：ST-GCN + 人口學（Stacking）", stack_probs, stack_target,
                    COLOR_POSITIVE, dict(lw=2.6)))

    per_action_a, labels_a = da.oof_probabilities("system_a_logreg.json", "combined")
    ids_a, probs_a, target_a = da.aligned(da.fuse(per_action_a), labels_a)
    sources.append(("系統 A：手工特徵 + 人口學", probs_a, target_a, COLOR_NEGATIVE,
                    dict(lw=2.2)))

    per_action_d, labels_d = da.oof_probabilities("system_a_logreg.json", "demographic")
    ids_d, probs_d, target_d = da.aligned(da.fuse(per_action_d), labels_d)
    sources.append(("基準線：只看年齡等人口學", probs_d, target_d, COLOR_NEUTRAL,
                    dict(lw=1.9, ls="--")))

    per_action_c, labels_c = da.oof_probabilities("system_shiftgcn_pretrained.json", "motion")
    ids_c, probs_c, target_c = da.aligned(da.fuse(per_action_c), labels_c)
    sources.append(("系統 C：Shift-GCN 純骨架", probs_c, target_c, COLOR_MUTED,
                    dict(lw=1.8, ls=":")))

    return sources


def roc_curves() -> None:
    """圖 12：ROC 曲線。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    ax = axes[0]
    for name, probs, target, color, kwargs in _curve_sources():
        fpr, tpr, _ = roc_curve(target, probs)
        auc = roc_auc_score(target, probs)
        ax.plot(fpr, tpr, color=color, label=f"{name}　AUC={auc:.3f}", **kwargs)

    ax.plot([0, 1], [0, 1], color="#C8D0D8", lw=1.2, ls="-")
    ax.text(0.55, 0.48, "隨機猜測 (AUC=0.5)", fontsize=8.4,
            color="#A6B0BA", rotation=38)
    ax.set_xlabel("偽陽性率（1 減特異度）")
    ax.set_ylabel("真陽性率（敏感度 / Recall）")
    ax.set_title("ROC 曲線　—　越靠左上角越好")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8.4)

    # 右：各動作
    ax = axes[1]
    per_action, labels = da.oof_probabilities("system_stgcn_pretrained.json", "motion")
    for action in ACTIONS:
        ids, probs, target = da.aligned(per_action[action], labels)
        fpr, tpr, _ = roc_curve(target, probs)
        auc = roc_auc_score(target, probs)
        ax.plot(fpr, tpr, color=ACTION_COLORS[action], lw=2.3,
                label=f"{action}　AUC={auc:.3f}")

    ids, probs, target = da.aligned(da.fuse(per_action), labels)
    fpr, tpr, _ = roc_curve(target, probs)
    ax.plot(fpr, tpr, color=COLOR_NEUTRAL, lw=2.8, ls="--",
            label=f"三動作融合　AUC={roc_auc_score(target, probs):.3f}")

    ax.plot([0, 1], [0, 1], color="#C8D0D8", lw=1.2)
    ax.set_xlabel("偽陽性率")
    ax.set_ylabel("真陽性率")
    ax.set_title("系統 B 各動作的 ROC　—　融合後優於任一單項")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8.6)

    fig.suptitle(
        "ROC 曲線（out-of-fold，31 位受試者）　—　"
        "曲線下面積即為 AUC",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    save(fig, "12_ROC曲線", "三系統與各動作")


def pr_curves() -> None:
    """圖 13：Precision-Recall 曲線——類別不平衡時比 ROC 更敏感。"""
    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    for name, probs, target, color, kwargs in _curve_sources():
        precision, recall, _ = precision_recall_curve(target, probs)
        ap = average_precision_score(target, probs)
        ax.plot(recall, precision, color=color, label=f"{name}　AP={ap:.3f}", **kwargs)

    baseline = float(np.mean(_curve_sources()[0][2]))
    ax.axhline(baseline, color="#C8D0D8", lw=1.2, ls="-")
    ax.text(0.02, baseline + 0.015,
            f"隨機猜測 = 陽性率 {baseline:.2f}", fontsize=8.4, color="#A6B0BA")

    ax.set_xlabel("Recall（敏感度）")
    ax.set_ylabel("Precision（精確率）")
    ax.set_title("Precision-Recall 曲線　—　陽性僅佔 39%，此圖比 ROC 更嚴格")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8.2)

    save(fig, "13_PR曲線", "類別不平衡下的補充指標")


def confusion_matrices() -> None:
    """圖 14：混淆矩陣——說明門檻選擇的臨床取捨。"""
    from src.evaluation import best_threshold

    per_action, labels = da.oof_probabilities("system_stgcn_pretrained.json", "motion")
    _, motion_probs, motion_target = da.aligned(da.fuse(per_action), labels)
    _, stack_probs, stack_target = da.stacking_oof("system_stgcn_pretrained.json")

    panels = [
        ("系統 B：純骨架", motion_probs, motion_target),
        ("系統 B：+ 人口學（Stacking）", stack_probs, stack_target),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    for ax, (title, probs, target) in zip(axes, panels):
        threshold = best_threshold(target, probs)
        predicted = (probs >= threshold).astype(int)
        tp = int(((predicted == 1) & (target == 1)).sum())
        fp = int(((predicted == 1) & (target == 0)).sum())
        tn = int(((predicted == 0) & (target == 0)).sum())
        fn = int(((predicted == 0) & (target == 1)).sum())
        matrix = np.array([[tn, fp], [fn, tp]])

        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=20)
        cells = [["正確排除", "誤報"], ["漏診", "正確偵測"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.12, str(matrix[i, j]), ha="center", va="center",
                        fontsize=21, fontweight="bold",
                        color="white" if matrix[i, j] > 10 else "#22303B")
                ax.text(j, i + 0.22, cells[i][j], ha="center", va="center",
                        fontsize=9,
                        color="white" if matrix[i, j] > 10 else "#4A5A66")

        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        ax.set_xticks([0, 1], ["判為陰性", "判為陽性"])
        ax.set_yticks([0, 1], ["實際無肌少症", "實際有肌少症"])
        ax.set_title(
            f"{title}（門檻 {threshold:.3f}）\n"
            f"敏感度 {recall:.3f}　特異度 {specificity:.3f}　精確率 {precision:.3f}",
            fontsize=10.5,
        )
        ax.grid(False)

    fig.suptitle(
        "混淆矩陣　—　右圖零誤報但漏診 5 人，左圖較適合篩檢用途",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    save(fig, "14_混淆矩陣", "門檻選擇的臨床取捨")


# ---------------------------------------------------------------- 比較

def pretraining_effect() -> None:
    """圖 15：合成資料預訓練的效果。"""
    baseline = da.load_result("system_stgcn.json")
    pretrained = da.load_result("system_stgcn_pretrained.json")

    items = list(ACTIONS) + ["三動作融合"]
    before, after = [], []
    for action in ACTIONS:
        before.append(baseline["groups"]["motion"][action]["summary"]["auc"])
        after.append(pretrained["groups"]["motion"][action]["summary"]["auc"])
    before.append(baseline["groups"]["motion"]["fusion"]["summary"]["auc"])
    after.append(pretrained["groups"]["motion"]["fusion"]["summary"]["auc"])

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(items))

    ax.bar(x - 0.19, before, 0.36, label="未預訓練", color=COLOR_MUTED)
    ax.bar(x + 0.19, after, 0.36, label="以合成資料預訓練", color=COLOR_ACCENT)

    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.6,
                color=COLOR_NEUTRAL)
        ax.text(i + 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.6,
                fontweight="bold", color=COLOR_ACCENT)
        ax.annotate(
            f"+{a - b:.3f}", xy=(i, max(a, b) + 0.055), ha="center",
            fontsize=9, fontweight="bold",
            color=COLOR_POSITIVE if a > b else COLOR_NEUTRAL,
        )

    ax.axhline(0.5, color="#C8D0D8", lw=1.2, ls="--")
    ax.text(len(items) - 0.45, 0.512, "隨機猜測", fontsize=8.2, color="#A6B0BA")
    ax.set_xticks(x, items)
    ax.set_ylabel("AUC（out-of-fold）")
    ax.set_ylim(0, 0.95)
    ax.set_title("合成資料預訓練的效果　—　融合後提升 0.228", pad=12)
    ax.legend(loc="upper left")
    annotate(ax,
             "證明骨架資料本身有訊號，\n"
             "先前學不到是因為 31 人不足以從零學起",
             "lower right")

    save(fig, "15_預訓練效果", "純骨架 0.561 → 0.789")


def action_ablation() -> None:
    """圖 16：動作組合消融——供收案決策參考。"""
    rows = da.load_result("ablation_system_stgcn_pretrained_motion.json")
    rows = sorted(rows, key=lambda r: r["summary"]["auc"])

    fig, ax = plt.subplots(figsize=(10, 6.4))
    positions = np.arange(len(rows))
    full = next(r for r in rows if r["n_actions"] == 3)["summary"]["auc"]

    for pos, row in zip(positions, rows):
        auc = row["summary"]["auc"]
        low, high = row["ci"]
        is_full = row["n_actions"] == 3
        color = COLOR_ACCENT if is_full else (
            COLOR_POSITIVE if auc > full else COLOR_MUTED
        )
        ax.barh(pos, auc, height=0.6, color=color, alpha=0.9)
        ax.plot([low, high], [pos, pos], color=COLOR_NEUTRAL, lw=1.5, zorder=3)
        ax.plot([low, low], [pos - 0.13, pos + 0.13], color=COLOR_NEUTRAL, lw=1.5)
        ax.plot([high, high], [pos - 0.13, pos + 0.13], color=COLOR_NEUTRAL, lw=1.5)
        ax.text(high + 0.012, pos, f"{auc:.3f}", va="center",
                fontsize=8.8, fontweight="bold" if is_full else "normal",
                color=COLOR_NEUTRAL)

    ax.axvline(full, color=COLOR_ACCENT, ls="--", lw=1.3, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([" + ".join(r["actions"]) for r in rows], fontsize=9.5)
    ax.set_xlabel("AUC（橫線為 95% 信賴區間）")
    ax.set_xlim(0.3, 1.05)
    ax.set_title("動作組合消融　—　信賴區間高度重疊，差異無統計意義", pad=12)
    annotate(ax,
             "唯一三系統一致的結論：\n行走（Walk）必須保留",
             "lower right")

    save(fig, "16_動作組合消融", "只做兩項時該做哪兩項")


def subject_predictions() -> None:
    """圖 17：逐人預測值——標出五個錯誤案例。"""
    from src.evaluation import best_threshold

    per_action, labels = da.oof_probabilities("system_stgcn_pretrained.json", "motion")
    ids, probs, target = da.aligned(da.fuse(per_action), labels)
    threshold = best_threshold(target, probs)
    profiles = da.subjects()

    order = np.argsort(probs)
    fig, ax = plt.subplots(figsize=(13, 6.2))

    for rank, idx in enumerate(order):
        group = target[idx]
        predicted = int(probs[idx] >= threshold)
        wrong = predicted != group
        ax.scatter(rank, probs[idx], s=150 if wrong else 88,
                   color=LABEL_COLORS[group], marker=LABEL_MARKERS[group],
                   edgecolor=COLOR_WARN if wrong else "white",
                   linewidth=2.2 if wrong else 1.0, zorder=3)
        if wrong:
            kind = "漏診" if group == 1 else "誤報"
            ax.annotate(
                f"{ids[idx]}\n{kind}\n{profiles[ids[idx]].age:.0f}歲",
                (rank, probs[idx]),
                xytext=(rank, probs[idx] + (0.16 if group == 1 else -0.20)),
                ha="center", fontsize=7.8, color=COLOR_WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLOR_WARN, lw=0.9),
            )

    ax.axhline(threshold, color=COLOR_NEUTRAL, ls="--", lw=1.4)
    ax.text(0.4, threshold + 0.018, f"判定門檻 {threshold:.3f}",
            fontsize=8.6, color=COLOR_NEUTRAL)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ids[i] for i in order], rotation=90, fontsize=7.4)
    ax.set_ylabel("預測風險機率")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("31 位受試者的預測值（依風險排序）　—　26 人正確、5 人錯誤", pad=12)
    ax.legend(handles=[
        Line2D([], [], marker=LABEL_MARKERS[k], ls="", color=LABEL_COLORS[k],
               label=f"實際{LABEL_NAMES[k]}", markersize=9) for k in (0, 1)
    ] + [
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=COLOR_WARN, markeredgewidth=2.2,
               label="預測錯誤", markersize=11)
    ], loc="upper left", ncol=3)
    annotate(ax,
             "誤報 2 人皆為男性（BMI 平均 27.9）\n漏診 3 人皆為女性（BMI 平均 22.6）",
             "lower right")

    save(fig, "17_逐人預測", "含五個錯誤案例標註")


def generate_all() -> None:
    print("【模型與結果】")
    architecture()
    roc_curves()
    pr_curves()
    confusion_matrices()
    pretraining_effect()
    action_ablation()
    subject_predictions()

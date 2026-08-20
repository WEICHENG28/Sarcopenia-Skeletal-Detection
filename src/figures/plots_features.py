"""特徵類圖表：臨床指標分布、判別力排行、降維視覺化。"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.dataset_manifest import ACTIONS
from src.figures import data_access as da
from src.figures.style import (
    ACTION_COLORS, COLOR_MUTED, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE,
    LABEL_COLORS, LABEL_MARKERS, LABEL_NAMES, annotate, save,
)

RANDOM_SEED = 42

# 臨床上最容易解釋的指標，供醫療人員判讀
CLINICAL_METRICS = {
    "Walk": [
        ("gait_speed_proxy", "步行速度估計", "越小越差"),
        ("cadence_hz", "步頻（步/秒）", "越小越差"),
        ("step_period_s", "步態週期（秒）", "越大越差"),
        ("knee_asymmetry", "左右膝不對稱", "越大越差"),
    ],
    "Sit": [
        ("sec_per_rep", "每次起立坐下（秒）", "越大越差"),
        ("knee_rom_deg", "膝關節活動範圍（度）", "—"),
        ("trunk_lean_max_deg", "軀幹最大前傾（度）", "越大越吃力"),
        ("standing_ratio", "站立時間佔比", "—"),
    ],
    "Squat": [
        ("descent_speed_deg_s", "下蹲速度（度/秒）", "越小越差"),
        ("hold_duration_s", "最低點停留（秒）", "越小越差"),
        ("knee_min_deg", "最低點膝角（度）", "越大代表蹲越淺"),
        ("squat_depth", "下蹲深度", "越小越差"),
    ],
}


def _metric_values(action: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    from src.features_handcrafted import build_feature_matrix

    matrix, labels, _, names = build_feature_matrix(list(da.manifest()), action)
    return matrix[:, names.index(key)], labels


def clinical_metrics() -> None:
    """圖 6：臨床指標在兩組間的分布。"""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5))

    for row, action in enumerate(ACTIONS):
        for col, (key, label, direction) in enumerate(CLINICAL_METRICS[action]):
            ax = axes[row, col]
            values, targets = _metric_values(action, key)
            auc = roc_auc_score(targets, values)

            for group in (0, 1):
                subset = values[targets == group]
                parts = ax.violinplot([subset], positions=[group],
                                      widths=0.66, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(LABEL_COLORS[group])
                    body.set_alpha(0.28)
                jitter = np.random.default_rng(row * 10 + col).normal(0, 0.05, len(subset))
                ax.scatter(group + jitter, subset, s=26,
                           color=LABEL_COLORS[group], marker=LABEL_MARKERS[group],
                           edgecolor="white", linewidth=0.6, zorder=3)
                ax.hlines(np.median(subset), group - 0.27, group + 0.27,
                          color=LABEL_COLORS[group], lw=2.2, zorder=4)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["無", "有"], fontsize=9)
            # 判別力以「距 0.5 多遠」表示，方向另以文字說明
            strength = max(auc, 1 - auc)
            ax.set_title(f"{label}\n判別力 {strength:.3f}", fontsize=9.8)
            if direction != "—":
                ax.set_xlabel(f"肌少症組{direction}", fontsize=8.2,
                              color=COLOR_NEUTRAL)
            if col == 0:
                ax.text(-0.32, 0.5, action, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold",
                        color=ACTION_COLORS[action])

    fig.suptitle(
        "臨床運動學指標的兩組分布　—　判別力 0.5 表示完全無法區分，越接近 1 越好",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    save(fig, "06_臨床指標分布", "三動作各四項指標")


def feature_ranking() -> None:
    """圖 7：所有動作特徵的判別力排行。"""
    from src.features_handcrafted import (
        DEMOGRAPHIC_FEATURES, QUALITY_FEATURES, build_feature_matrix,
    )

    excluded = set(DEMOGRAPHIC_FEATURES) | set(QUALITY_FEATURES)
    rows: List[Tuple[float, str, str, bool]] = []

    for action in ACTIONS:
        matrix, labels, _, names = build_feature_matrix(list(da.manifest()), action)
        for i, name in enumerate(names):
            if name in excluded or np.ptp(matrix[:, i]) < 1e-9:
                continue
            auc = roc_auc_score(labels, matrix[:, i])
            higher_is_positive = auc > 0.5
            rows.append((max(auc, 1 - auc), action, name, higher_is_positive))

    rows.sort(reverse=True)
    rows = rows[:18]

    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    positions = np.arange(len(rows))[::-1]

    for pos, (strength, action, name, higher) in zip(positions, rows):
        ax.barh(pos, strength - 0.5, left=0.5, height=0.66,
                color=ACTION_COLORS[action], alpha=0.88)
        arrow = "↑ 肌少症組較高" if higher else "↓ 肌少症組較低"
        ax.text(strength + 0.008, pos, f"{strength:.3f}　{arrow}",
                va="center", fontsize=8.4, color=COLOR_NEUTRAL)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{a} · {n}" for _, a, n, _ in rows], fontsize=8.8)
    ax.set_xlim(0.5, 0.92)
    ax.set_xlabel("單一特徵的判別力（AUC 距 0.5 的絕對距離）")
    ax.axvline(0.5, color=COLOR_NEUTRAL, lw=1.0)
    ax.set_title("動作特徵判別力排行（前 18 名）", pad=12)
    ax.legend(handles=[
        Line2D([], [], color=ACTION_COLORS[a], lw=8, label=a) for a in ACTIONS
    ], loc="lower right")
    annotate(ax,
             "作為對照：年齡單獨的判別力為 0.879\n"
             "目前沒有任何單一動作特徵超越年齡",
             "upper right")

    fig.tight_layout()
    save(fig, "07_特徵判別力排行", "最強動作特徵為 Walk 行走耗時")


def _embed(matrix: np.ndarray, method: str, seed: int = RANDOM_SEED) -> np.ndarray:
    """把高維特徵降到二維。"""
    scaled = StandardScaler().fit_transform(matrix)

    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(scaled)
    if method == "t-SNE":
        # perplexity 必須小於樣本數；31 人下取 8 較穩定
        return TSNE(n_components=2, perplexity=8, init="pca",
                    learning_rate="auto", random_state=seed).fit_transform(scaled)
    if method == "Isomap":
        return Isomap(n_components=2, n_neighbors=8).fit_transform(scaled)
    if method == "UMAP":
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return umap.UMAP(n_components=2, n_neighbors=8, min_dist=0.3,
                             random_state=seed).fit_transform(scaled)
    raise ValueError(method)


def _scatter_embedding(ax, points: np.ndarray, labels: np.ndarray,
                       ids: List[str], title: str) -> None:
    for group in (0, 1):
        mask = labels == group
        ax.scatter(points[mask, 0], points[mask, 1],
                   s=95, color=LABEL_COLORS[group], marker=LABEL_MARKERS[group],
                   edgecolor="white", linewidth=1.2, alpha=0.9,
                   label=f"{LABEL_NAMES[group]}（{mask.sum()} 人）", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def dimension_reduction(matrix: np.ndarray, labels: np.ndarray, ids: List[str],
                        name: str, subtitle: str, note: str) -> None:
    """以四種降維方法呈現特徵的可分性。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4))
    methods = ["PCA", "t-SNE", "UMAP", "Isomap"]

    for ax, method in zip(axes.ravel(), methods):
        points = _embed(matrix, method)
        _scatter_embedding(ax, points, labels, ids, method)

    axes[0, 0].legend(loc="best")
    fig.suptitle(subtitle, fontsize=13, y=1.0)
    fig.text(0.5, -0.01, note, ha="center", fontsize=9,
             color=COLOR_NEUTRAL, style="italic")
    fig.tight_layout()
    save(fig, name, "PCA / t-SNE / UMAP / Isomap")


def dimension_reduction_handcrafted() -> None:
    """圖 8：手工特徵的降維——無模型介入，反映資料本身的可分性。"""
    ids, matrix, labels, _ = da.handcrafted_features()
    dimension_reduction(
        matrix, labels, ids,
        name="08_降維_手工特徵",
        subtitle=(
            "手工運動學特徵的二維投影　—　兩類明顯重疊，"
            "顯示單憑動作特徵難以完全分開"
        ),
        note=(
            "特徵為確定性計算，未經任何模型訓練，因此無資料洩漏疑慮；"
            "此為「資料本身可分性」最誠實的呈現。"
        ),
    )


def dimension_reduction_stgcn() -> None:
    """圖 9：ST-GCN 學到的 16 維時空特徵。"""
    ids, features, labels = da.stgcn_features()
    matrix = np.hstack([features[a] for a in ACTIONS])
    valid = ~np.isnan(matrix).any(axis=1)

    dimension_reduction(
        matrix[valid], labels[valid], [i for i, v in zip(ids, valid) if v],
        name="09_降維_STGCN特徵",
        subtitle=(
            "ST-GCN 時空特徵向量的二維投影（3 動作 × 16 維 = 48 維）　—　"
            "分離程度明顯優於手工特徵"
        ),
        note=(
            "【注意】此處使用的最終模型以全部 31 位個案訓練，特徵帶有記憶成分。"
            "分離良好不代表對新個案的判別能力；實際效能請見圖 12 的 ROC 曲線（out-of-fold）。"
        ),
    )


def dimension_reduction_oof() -> None:
    """圖 10：out-of-fold 預測空間——這才是真正的判別力。"""
    per_action, labels_map = da.oof_probabilities(
        "system_stgcn_pretrained.json", "motion"
    )
    common = sorted(set.intersection(*[set(v) for v in per_action.values()]))
    matrix = np.array([[per_action[a][s] for a in ACTIONS] for s in common])
    labels = np.array([labels_map[s] for s in common])
    profiles = da.subjects()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        for group in (0, 1):
            mask = labels == group
            ax.scatter(matrix[mask, i], matrix[mask, j],
                       s=95, color=LABEL_COLORS[group], marker=LABEL_MARKERS[group],
                       edgecolor="white", linewidth=1.2, alpha=0.9,
                       label=LABEL_NAMES[group], zorder=3)
        ax.axhline(0.5, color=COLOR_MUTED, ls=":", lw=1.0)
        ax.axvline(0.5, color=COLOR_MUTED, ls=":", lw=1.0)
        ax.set_xlabel(f"{ACTIONS[i]} 預測機率")
        ax.set_ylabel(f"{ACTIONS[j]} 預測機率")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)

    axes[0].legend(loc="upper left")
    fig.suptitle(
        "三個動作的 out-of-fold 預測機率兩兩對照　—　"
        "右上角為三項皆判為高風險，左下角為皆判為低風險",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "10_預測空間", "無洩漏的真實判別情形")


def generate_all() -> None:
    print("【特徵與降維】")
    clinical_metrics()
    feature_ranking()
    dimension_reduction_handcrafted()
    dimension_reduction_stgcn()
    dimension_reduction_oof()

"""圖表共用樣式。

設計原則：
- 供醫療人員閱讀，因此每張圖都需自我說明（標題點出結論，而非只寫欄位名）
- 色彩需在黑白列印與色盲條件下仍可辨識，故形狀與顏色雙重編碼
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# 語意色：陰性=藍綠、陽性=橘紅。避免紅綠對比（紅綠色盲最常見）
COLOR_NEGATIVE = "#0F9D8F"   # 無肌少症
COLOR_POSITIVE = "#E8603C"   # 有肌少症
COLOR_NEUTRAL = "#5B6B7A"
COLOR_ACCENT = "#3D6FB4"
COLOR_MUTED = "#B8C2CC"
COLOR_WARN = "#D9A441"

# 三個動作固定配色，跨圖一致
ACTION_COLORS = {
    "Sit": "#3D6FB4",
    "Walk": "#0F9D8F",
    "Squat": "#B4699B",
}

LABEL_NAMES = {0: "無肌少症", 1: "肌少症"}
LABEL_COLORS = {0: COLOR_NEGATIVE, 1: COLOR_POSITIVE}
LABEL_MARKERS = {0: "o", 1: "^"}   # 形狀輔助，黑白列印仍可分辨


def setup() -> None:
    """套用全域樣式，並設定中文字型。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft JhengHei", "Microsoft YaHei", "DFKai-SB", "SimHei"):
        if candidate in available:
            # DejaVu Sans 補上中文字型缺少的數學符號（≥、−）
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCD3DA",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8ECF0",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save(fig, name: str, caption: str = "") -> Path:
    """存檔並回傳路徑。caption 會印出，供撰寫報告時複製。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}" + (f"　{caption}" if caption else ""))
    return path


def annotate(ax, text: str, loc: str = "lower right") -> None:
    """在圖上加註說明文字。"""
    positions = {
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha=ha, va=va, fontsize=8.5, color=COLOR_NEUTRAL,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#DDE3E9", alpha=0.92),
    )

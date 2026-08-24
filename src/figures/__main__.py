"""產生全部圖表。

執行：
    python -m src.figures            # 全部
    python -m src.figures --only 資料 # 只跑資料類

產出至 figures/，並自動更新 figures/README.md 索引。
"""

from __future__ import annotations

import argparse

from src.figures import plots_data, plots_features, plots_model, plots_summary
from src.figures.style import FIGURES_DIR, setup

GROUPS = {
    "資料": plots_data,
    "特徵": plots_features,
    "模型": plots_model,
    "總結": plots_summary,
}


def write_index() -> None:
    """依實際產出的檔案更新索引。"""
    sections = {
        "資料理解": [
            ("01_骨架點位定義", "MediaPipe 33 點的位置，以及本研究採用的身體 22 點"),
            ("02_三動作姿勢", "起立坐下／行走／半蹲的代表性骨架姿勢"),
            ("03_年齡混淆", "兩組平均年齡相差 13 歲，是本研究最大的干擾因子"),
            ("04_資料品質", "各關節的可見度；行走是唯一需改善拍攝角度的動作"),
            ("05_序列長度分布", "三動作長度差 3 倍，因此各自設定模型輸入長度"),
        ],
        "特徵與可分性": [
            ("06_臨床指標分布", "三動作各四項臨床指標在兩組間的分布"),
            ("07_特徵判別力排行", "所有動作特徵的判別力，最強為行走耗時"),
            ("08_降維_手工特徵", "PCA / t-SNE / UMAP / Isomap；無模型介入，反映資料本身可分性"),
            ("09_降維_STGCN特徵", "ST-GCN 學到的 48 維時空特徵（含記憶成分，見圖說）"),
            ("10_預測空間", "三動作 out-of-fold 預測機率的兩兩對照"),
        ],
        "模型與結果": [
            ("11_分類器架構", "從骨架序列到風險分級的完整架構"),
            ("12_ROC曲線", "三系統與各動作的 ROC 曲線"),
            ("13_PR曲線", "Precision-Recall 曲線（類別不平衡下更嚴格）"),
            ("14_混淆矩陣", "門檻選擇的臨床取捨：零誤報 vs 少漏診"),
            ("15_預訓練效果", "合成資料預訓練使純骨架 AUC 由 0.561 升至 0.789"),
            ("16_動作組合消融", "只做兩項動作時該做哪兩項"),
            ("17_逐人預測", "31 位受試者的預測值，標出五個錯誤案例"),
        ],
        "總結": [
            ("18_系統流程", "從手機影片到風險報告的六個階段"),
            ("19_信賴區間比較", "所有系統的信賴區間高度重疊，無法宣稱誰較優"),
            ("20_半蹲維持期發現", "臨床協定決定了該測量什麼"),
            ("21_三系統比較", "判斷模型是否真有貢獻，必須看「純骨架」欄"),
        ],
    }

    lines = [
        "# 圖表索引",
        "",
        "由 `python -m src.figures` 自動產生。所有數值取自 out-of-fold 交叉驗證，",
        "與 [RESULTS.md](../RESULTS.md) 一致。",
        "",
        "> **給醫療人員**：圖 18（系統流程）、圖 03（年齡混淆）、圖 04（資料品質）、",
        "> 圖 20（半蹲維持期）最能快速說明本研究的重點與限制。",
        "",
        "## 下載 PDF",
        "",
        "全部圖表已彙整為單一 PDF，含封面、目錄與逐圖說明，方便列印與傳閱：",
        "",
        "**[肌少症偵測系統_圖表報告.pdf](肌少症偵測系統_圖表報告.pdf)**（28 頁）",
        "",
        "重新產生：`python -m src.figures.report`",
        "",
    ]

    for section, items in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        for name, description in items:
            path = FIGURES_DIR / f"{name}.png"
            if not path.exists():
                continue
            lines.append(f"### {name.split('_', 1)[1].replace('_', ' ')}")
            lines.append("")
            lines.append(f"{description}")
            lines.append("")
            lines.append(f"![{name}]({name}.png)")
            lines.append("")

    lines += [
        "---",
        "",
        "## 重要提醒",
        "",
        "1. **圖 09（ST-GCN 特徵降維）的分離程度不代表判別能力**——該圖使用以全部 31 人訓練的最終模型，",
        "   特徵帶有記憶成分。真實效能請看圖 12（ROC，out-of-fold）。",
        "2. **圖 19 是解讀所有結果的前提**——31 人的樣本量下，各系統的信賴區間高度重疊，",
        "   不足以宣稱任一系統優於「只看年齡」的基準線。",
        "3. **圖 12 的 AUC 高不等於臨床上好用**——見圖 14，AUC 最高的設定漏診最多。",
        "",
    ]

    (FIGURES_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n索引已更新：{FIGURES_DIR / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="產生專案圖表")
    parser.add_argument("--only", choices=list(GROUPS), help="只產生指定分類")
    args = parser.parse_args()

    setup()
    targets = {args.only: GROUPS[args.only]} if args.only else GROUPS
    for module in targets.values():
        module.generate_all()

    write_index()
    print(f"全部圖表位於：{FIGURES_DIR}")


if __name__ == "__main__":
    main()

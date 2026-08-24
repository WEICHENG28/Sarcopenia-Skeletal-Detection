"""把圖表彙整成單一 PDF，方便下載與傳閱。

以 A4 橫式排版，每張圖獨立一頁並附說明文字。
不需安裝 LaTeX 或其他轉檔工具。

執行：
    python -m src.figures.report
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.figures.style import (
    COLOR_ACCENT, COLOR_MUTED, COLOR_NEUTRAL, COLOR_POSITIVE, FIGURES_DIR, setup,
)

# A4 橫式（英吋）
PAGE = (11.69, 8.27)

OUTPUT = FIGURES_DIR / "肌少症偵測系統_圖表報告.pdf"

# (章節, [(檔名, 標題, 說明)])
SECTIONS: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    (
        "一、資料理解",
        "先看清楚手上的資料長什麼樣、有什麼限制",
        [
            ("01_骨架點位定義", "骨架關鍵點定義",
             "MediaPipe Pose 對每一幀影像輸出 33 個關鍵點。本研究只採用身體的 22 點"
             "（編號 11–32），捨棄臉部的 11 點。\n\n"
             "捨棄的原因不是臉部不重要，而是在 MediaPipe 的官方骨架連線定義中，"
             "臉部與身體並不相連。圖神經網路依靠關節之間的「連線」傳遞資訊，"
             "不相連的部位會讓整張圖分裂成三塊互不相通的子圖，"
             "臉部的資訊永遠無法傳到身體，反而成為雜訊。"),
            ("02_三動作姿勢", "三項檢測動作",
             "起立坐下、向前行走、半蹲。三者皆為 AWGS 2019（亞洲肌少症工作小組）"
             "診斷準則中「體能表現」的正式測量項目。\n\n"
             "本系統要做的，是把原本需要治療師拿碼錶手動計時的項目自動化。"),
            ("03_年齡混淆", "年齡是最大的干擾因子",
             "肌少症組的平均年齡比對照組大 13 歲（80.0 歲 vs 74.1 歲），"
             "80 歲以上的 8 人中有 7 人為陽性。\n\n"
             "這代表「只看年齡」不做任何動作分析，判別力就已經達到 AUC 0.879。"
             "因此本研究的所有結果都必須拆成三種對照：只看年齡、只看動作、兩者合併，"
             "否則無法分辨模型究竟是在分析動作，還是只是在看年齡。\n\n"
             "**建議後續收案優先補收：80 歲以上的健康長者、75 歲以下的肌少症患者。**"),
            ("04_資料品質", "錄影品質稽核",
             "MediaPipe 在關節被遮擋時，仍會「推測」並輸出一組外觀完全正常的座標，"
             "因此低品質的資料從數值上看不出異常。\n\n"
             "三項動作中，只有**行走**的下肢可見度偏低（0.916，遮擋幀佔 7.7%），"
             "個別樣本更低至 0.504。原因是側面拍攝時，靠鏡頭側的肢體會遮住另一側。\n\n"
             "**建議：行走改以斜前方 45 度拍攝，或加設第二機位。**"
             "此項改善對資料品質的效益高於單純增加收案人數。"),
            ("05_序列長度分布", "動作長度分布",
             "三個動作的長度差距達 3 倍（起立坐下中位數 14 秒、行走 5 秒、半蹲 6 秒），"
             "因此模型的輸入長度必須各自設定。\n\n"
             "早期版本三個動作共用 3.2 秒的輸入長度，會把 14 秒的起立坐下大幅截斷。"),
        ],
    ),
    (
        "二、特徵與可分性",
        "資料裡究竟有沒有可用的訊號？",
        [
            ("06_臨床指標分布", "臨床運動學指標",
             "由骨架資料自動計算的臨床指標，在兩組間的分布。"
             "「判別力」為 AUC 距 0.5 的距離：0.5 表示完全無法區分，越接近 1 越好。\n\n"
             "行走的多項指標（步速、步頻、步態週期）方向一致地指向"
             "「肌少症組走得較慢」，與 AWGS 2019 以步速為核心診斷條件相符。"),
            ("07_特徵判別力排行", "特徵判別力排行",
             "所有動作特徵的判別力排序。最強者為行走耗時（0.798）。\n\n"
             "作為對照：年齡單獨的判別力為 0.879。"
             "**目前沒有任何單一動作特徵能超越年齡。**"),
            ("08_降維_手工特徵", "手工特徵的二維投影",
             "以四種降維方法（PCA、t-SNE、UMAP、Isomap）把多維特徵壓縮到平面上觀察。\n\n"
             "兩類明顯重疊，代表單憑這些人工設計的動作指標難以完全區分兩組。\n\n"
             "**這張圖的可信度最高**——特徵為確定性的數學計算，未經任何模型訓練，"
             "因此不存在資料洩漏的疑慮，是「資料本身可分性」最誠實的呈現。"),
            ("09_降維_STGCN特徵", "ST-GCN 特徵的二維投影",
             "ST-GCN 學到的時空特徵向量（3 動作 × 16 維 = 48 維）的二維投影。\n\n"
             "**注意：分離良好不代表判別能力。**"
             "此圖使用的最終模型是以全部 31 位個案訓練的，模型已經「看過」這些人，"
             "特徵帶有記憶成分。分得開是理所當然的結果。\n\n"
             "**若要說明模型的真實效能，請引用 ROC 曲線（下一章），而非此圖。**"),
            ("10_預測空間", "out-of-fold 預測空間",
             "三個動作的預測機率兩兩對照。每個人的預測值都來自「沒有看過他」的模型，"
             "因此不含記憶成分，反映真實的判別情形。\n\n"
             "右上角為三項動作皆判為高風險，左下角為皆判為低風險。"),
        ],
    ),
    (
        "三、模型與結果",
        "模型怎麼做判斷，做得如何",
        [
            ("11_分類器架構", "分類器架構",
             "分為兩階段：\n\n"
             "**階段一**——三個動作各有一個獨立的 ST-GCN 模型。"
             "骨架序列經過兩層圖卷積後，壓縮成 16 個數字（時空特徵向量），"
             "再由 Linear 層加權相加成一個分數，最後以 Sigmoid 壓成 0~1 的機率。\n\n"
             "**階段二**——三個動作的機率先加權平均（Late Fusion），"
             "再與年齡等基本資料一起交給 Logistic Regression 做最終判斷。\n\n"
             "分成兩階段的原因：若把骨架特徵與年齡丟進同一個網路聯合訓練，"
             "16 維的骨架特徵會讓模型有足夠自由度去「背下」訓練樣本，"
             "反而學不到「年齡越大風險越高」這種簡單關係。"),
            ("12_ROC曲線", "ROC 曲線",
             "曲線下的面積即為 AUC。曲線越靠左上角，代表在相同誤報率下能抓出越多真陽性。\n\n"
             "所有曲線皆使用 out-of-fold 預測（每個人都由沒看過他的模型評估），"
             "數值與研究報告完全一致。\n\n"
             "**重點**：系統 B 的純骨架版本（0.789）在完全不使用年齡的情況下，"
             "已與「只看年齡」的基準線（0.794）相當。"),
            ("13_PR曲線", "Precision-Recall 曲線",
             "陽性個案僅佔 39%，屬於類別不平衡的情境。"
             "此時 PR 曲線比 ROC 更為嚴格，可避免高估模型表現。"),
            ("14_混淆矩陣", "混淆矩陣與門檻取捨",
             "左圖（純骨架）：抓出 9 位陽性、誤報 2 位、漏診 3 位。\n"
             "右圖（加入年齡）：零誤報，但漏診增加到 5 位。\n\n"
             "**AUC 較高的設定，反而漏診較多。**\n\n"
             "篩檢工具通常寧可誤報也不能漏診，因此左圖的設定在臨床上較為適用。"
             "**最終門檻應由臨床端依可接受的漏診率決定，而非取 AUC 最高者。**"),
            ("15_預訓練效果", "合成資料預訓練的效果",
             "31 位真實個案不足以讓深度模型從零學起。"
             "先以 58 位模擬個案預訓練（僅使用骨架，不使用其年齡等欄位），"
             "再以真實個案微調後，純骨架的判別力由 0.561 提升至 0.789。\n\n"
             "**這是本研究最主要的發現：骨架資料本身確實帶有訊號，"
             "先前學不到並非方法有誤，而是樣本量不足。**"),
            ("16_動作組合消融", "動作組合消融",
             "回答「只能做兩項動作時該做哪兩項」。\n\n"
             "橫線為 95% 信賴區間。各組合的區間高度重疊，"
             "代表在 31 人的樣本量下，這些差距沒有統計意義。\n\n"
             "**目前唯一能確定的是：行走必須保留。**"),
            ("17_逐人預測", "逐人預測與錯誤案例",
             "31 人中 26 人判斷正確、5 人錯誤。\n\n"
             "錯誤呈現兩個值得注意的趨勢：\n"
             "・誤報的 2 人皆為男性、體型較壯（BMI 平均 27.9）\n"
             "・漏診的 3 人皆為女性、體型偏瘦（BMI 平均 22.6）\n\n"
             "若此趨勢屬實則為危險偏誤——瘦小的長者正是肌少症的高風險族群。"
             "樣本數過少尚無法斷定，但方向值得後續追蹤。\n\n"
             "三位漏診者的動作機率全面偏低，推測為症狀較輕的早期個案，"
             "或是「動作尚可但握力已下降」的類型（握力為模型無法取得的資訊）。"),
        ],
    ),
    (
        "四、總結",
        "整體流程與必須誠實說明的限制",
        [
            ("18_系統流程", "系統整體流程",
             "從一支手機影片，到一份可解釋的風險報告，共六個階段。\n\n"
             "下方標註了四個關鍵的設計決定，包括為何使用 world landmarks、"
             "如何處理不同場地的鏡頭方向不一致、"
             "以及為何患者僅完成 2/3 項動作時仍可評估。"),
            ("19_信賴區間比較", "信賴區間比較",
             "**這是解讀所有結果的前提。**\n\n"
             "六個系統的 95% 信賴區間高度重疊，且所有估計值都落在"
             "基準線（只看年齡）的信賴區間 [0.59, 0.95] 之內。\n\n"
             "**以 31 人的樣本量，尚不足以宣稱任何模型優於「只看年齡」。**"
             "本研究屬於可行性研究等級，證明方向可行，但非經臨床驗證的醫療器材。\n\n"
             "信賴區間會隨樣本數增加而收窄，是最直接的進度指標。"),
            ("20_半蹲維持期發現", "半蹲：臨床協定決定了該測量什麼",
             "初版的半蹲特徵只測量「蹲多深」，模型判別力僅 0.465（低於隨機），"
             "當時的結論是建議移除此動作。\n\n"
             "向物理治療師確認後才得知，協定的核心是"
             "「蹲到最低點**停留約 3 秒**」——若不要求停留時間，"
             "長者容易蹲下即刻起身或反覆快速上下，兩者皆有跌倒風險。\n\n"
             "補上維持期的特徵後，判別力提升至 0.737，先前的移除建議因此撤回。\n\n"
             "**若當初僅依模型分數就移除半蹲，將永久失去一個具臨床價值的檢測項目。**"
             "這說明 AI 無法取代臨床知識，兩者必須並行。\n\n"
             "另外，僅 8/30 人達到協定要求的 3 秒。但撐不到 2 秒者以肌少症居多"
             "（13 人中 8 位），代表「撐不住」本身即為有效訊號，並非收案失誤。"),
            ("21_三系統比較", "三系統比較",
             "**判斷模型是否真有貢獻，必須看「純骨架」欄。**\n\n"
             "系統 C（Shift-GCN）的合併分數 0.828 看似最高，"
             "但其純骨架表現僅 0.439（低於隨機猜測），"
             "代表這個分數幾乎完全來自年齡，圖神經網路沒有實質貢獻。\n\n"
             "若只比較合併欄位，會得出「Shift-GCN 最好」的錯誤結論。"),
        ],
    ),
]


def _blank_page(pdf: PdfPages) -> plt.Figure:
    fig = plt.figure(figsize=PAGE)
    return fig


def _cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.70, "肌少症骨架偵測系統", ha="center",
             fontsize=30, fontweight="bold", color="#22303B")
    fig.text(0.5, 0.62, "圖表分析報告", ha="center",
             fontsize=19, color=COLOR_ACCENT)

    fig.add_artist(plt.Line2D([0.30, 0.70], [0.575, 0.575],
                              color=COLOR_ACCENT, lw=2.2))

    summary = (
        "以 MediaPipe 骨架資料與圖神經網路，"
        "自動化 AWGS 2019 體能表現評估\n\n"
        "受試者 31 人（肌少症 12 人 / 無肌少症 19 人）　·　動作樣本 92 筆\n"
        "評估方式：5-fold 交叉驗證，以受試者分組，全數採 out-of-fold 預測"
    )
    fig.text(0.5, 0.46, summary, ha="center", va="top",
             fontsize=11.5, color=COLOR_NEUTRAL, linespacing=1.9)

    caution = (
        "本報告為可行性研究成果，非經臨床驗證之醫療器材。\n"
        "所有評估結果僅供篩檢參考，不可取代專業醫療診斷。"
    )
    fig.text(0.5, 0.235, caution, ha="center", va="top",
             fontsize=10.5, color=COLOR_POSITIVE, linespacing=1.9,
             bbox=dict(boxstyle="round,pad=0.9", facecolor="#FDF3F0",
                       edgecolor="#F3D4CB"))

    fig.text(0.5, 0.08, f"產出日期：{date.today().isoformat()}",
             ha="center", fontsize=9.5, color=COLOR_MUTED)

    pdf.savefig(fig)
    plt.close(fig)


def _contents(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.91, "目錄", fontsize=22, fontweight="bold", color="#22303B")

    y = 0.83
    number = 1
    for section, subtitle, items in SECTIONS:
        fig.text(0.08, y, section, fontsize=13.5,
                 fontweight="bold", color=COLOR_ACCENT)
        fig.text(0.30, y, subtitle, fontsize=10, color=COLOR_NEUTRAL)
        y -= 0.043
        for _, title, _ in items:
            fig.text(0.11, y, f"圖 {number:02d}", fontsize=10,
                     color=COLOR_MUTED)
            fig.text(0.185, y, title, fontsize=10.5, color="#3A4750")
            y -= 0.034
            number += 1
        y -= 0.018

    fig.text(0.08, 0.055,
             "建議給醫療人員閱讀的四張：圖 18（系統流程）、圖 03（年齡混淆）、"
             "圖 04（資料品質）、圖 20（半蹲維持期）",
             fontsize=9.5, color=COLOR_NEUTRAL, style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def _section_divider(pdf: PdfPages, section: str, subtitle: str) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("#F7F9FB")
    fig.text(0.5, 0.54, section, ha="center", fontsize=27,
             fontweight="bold", color=COLOR_ACCENT)
    fig.text(0.5, 0.46, subtitle, ha="center", fontsize=13,
             color=COLOR_NEUTRAL)
    pdf.savefig(fig, facecolor="#F7F9FB")
    plt.close(fig)


def _display_width(text: str) -> float:
    """估算字串寬度：中日韓字元佔兩個半形位。"""
    return sum(2.0 if ord(c) > 0x2E7F else 1.0 for c in text)


def _wrap(text: str, limit: float) -> List[str]:
    """中文斷行。

    中文沒有空格可供斷點，matplotlib 內建的 wrap 會在版面邊緣硬切，
    導致標點落在行首或英文單字被切斷，因此自行處理。
    """
    # 避免行首出現這些標點
    forbidden_leading = "，。、；：）」』？！%"
    lines: List[str] = []

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue

        current = ""
        for char in paragraph:
            if _display_width(current + char) > limit and char not in forbidden_leading:
                lines.append(current)
                current = char
            else:
                current += char
        if current:
            lines.append(current)

    return lines


def _figure_page(pdf: PdfPages, name: str, number: int,
                 title: str, body: str) -> bool:
    """圖與說明同頁：閱讀時不需在兩頁之間來回翻找。"""
    path = FIGURES_DIR / f"{name}.png"
    if not path.exists():
        print(f"  ! 找不到 {path.name}，略過")
        return False

    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.955, f"圖 {number:02d}　{title}",
             fontsize=15, fontweight="bold", color="#22303B")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.937, 0.937],
                              color="#DDE3E9", lw=1.2))

    # 圖片依原始長寬比置中，避免變形
    image = mpimg.imread(path)
    height, width = image.shape[:2]
    area_w, area_h = 0.90, 0.56
    page_ratio = PAGE[0] / PAGE[1]
    scale = min(area_w / (width / height / page_ratio), area_h)
    box_h = scale
    box_w = box_h * (width / height) / page_ratio

    ax = fig.add_axes([0.5 - box_w / 2, 0.345, box_w, box_h])
    ax.imshow(image)
    ax.axis("off")

    # 說明區
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.315, 0.315],
                              color="#DDE3E9", lw=1.0))
    fig.text(0.05, 0.288, "說明", fontsize=10.5,
             fontweight="bold", color=COLOR_ACCENT)

    y = 0.252
    for paragraph in body.split("\n\n"):
        emphasised = paragraph.startswith("**") and paragraph.rstrip().endswith("**")
        clean = paragraph.replace("**", "").strip()
        if not clean:
            continue
        for line in _wrap(clean, limit=104):
            if y < 0.035:
                break
            fig.text(0.05, y, line, fontsize=9.6, va="top", ha="left",
                     color="#22303B" if emphasised else "#3A4750",
                     fontweight="bold" if emphasised else "normal")
            y -= 0.0245
        y -= 0.010

    pdf.savefig(fig)
    plt.close(fig)
    return True


def _closing(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.91, "解讀本報告時的三項提醒",
             fontsize=20, fontweight="bold", color="#22303B")

    notes = [
        ("一、圖 09 的分離程度不代表判別能力",
         "該圖使用以全部 31 位個案訓練的最終模型，特徵帶有記憶成分。"
         "模型「看過」這些人，分得開是理所當然。\n"
         "若要說明真實效能，請引用圖 12 的 ROC 曲線（out-of-fold）。"),
        ("二、圖 19 是解讀所有結果的前提",
         "31 人的樣本量下，各系統的 95% 信賴區間高度重疊，"
         "且都涵蓋「只看年齡」的基準線 0.794。\n"
         "本研究不足以宣稱任何模型優於單看年齡，屬可行性研究等級。"),
        ("三、AUC 高不等於臨床上好用",
         "見圖 14：AUC 最高的設定雖然零誤報，卻漏診 5 位（共 12 位陽性）。\n"
         "篩檢工具通常寧可誤報也不能漏診，最終門檻應由臨床端決定。"),
    ]

    y = 0.79
    for heading, body in notes:
        fig.text(0.08, y, heading, fontsize=13.5,
                 fontweight="bold", color=COLOR_ACCENT)
        fig.text(0.08, y - 0.045, body, fontsize=11.5, va="top",
                 color="#3A4750", linespacing=1.95)
        y -= 0.215

    fig.text(0.08, 0.13,
             "本報告由 python -m src.figures.report 自動產生，"
             "所有數值與專案的 RESULTS.md 一致。",
             fontsize=9.5, color=COLOR_MUTED, style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def build(output: Path = OUTPUT) -> Path:
    setup()
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        _cover(pdf)
        _contents(pdf)

        number = 1
        for section, subtitle, items in SECTIONS:
            _section_divider(pdf, section, subtitle)
            for name, title, body in items:
                _figure_page(pdf, name, number, title, body)
                number += 1

        _closing(pdf)

        info = pdf.infodict()
        info["Title"] = "肌少症骨架偵測系統 — 圖表分析報告"
        info["Subject"] = "以 MediaPipe 骨架與 ST-GCN 進行肌少症風險篩檢的可行性研究"
        info["Keywords"] = "肌少症, MediaPipe, ST-GCN, AWGS 2019, 篩檢"

    size_mb = output.stat().st_size / 1e6
    print(f"PDF 已產生：{output}（{size_mb:.1f} MB）")
    return output


if __name__ == "__main__":
    build()

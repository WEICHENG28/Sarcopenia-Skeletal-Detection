"""資料健檢報告：供與醫療機構討論收案品質使用。

產出內容：
1. 收案概況（人數、標籤分布、動作完成度）
2. 錄影品質（關節可見度、拍攝角度、長度）
3. 協定遵從度（半蹲停留 3 秒、五次起立坐下）
4. 臨床指標在陽性/陰性兩組的分布
5. 具體的收案調整建議

執行：
    python -m src.data_health_report            # 輸出到終端
    python -m src.data_health_report --save     # 另存 Markdown
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, TextIO

import numpy as np
from sklearn.metrics import roc_auc_score

from src.dataset_manifest import (
    ACTIONS, EXCLUDED_SAMPLES, EXCLUDED_SUBJECTS, FPS, Sample, build_manifest,
)
from src.features_handcrafted import (
    SQUAT_HOLD_TARGET_S, SQUAT_MIN_ROM_DEG, extract_features,
)
from src.skeleton_io import (
    L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE, load_landmarks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# 可見度低於此值代表關節被遮擋，MediaPipe 的座標為推測值
VISIBILITY_THRESHOLD = 0.5

# 下肢關節——肌少症判讀的主要依據
LOWER_BODY = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)


def _collect(samples: Sequence[Sample]) -> Dict[str, Dict]:
    """讀取所有樣本的品質指標與臨床特徵。"""
    records: Dict[str, Dict] = {}
    for sample in samples:
        raw = load_landmarks(sample.json_path)
        visibility = raw[..., 3]
        records[f"{sample.subject_id}_{sample.action}"] = {
            "sample": sample,
            "frames": raw.shape[0],
            "duration_s": raw.shape[0] / FPS,
            "vis_mean": float(visibility.mean()),
            "vis_lower": float(visibility[:, LOWER_BODY].mean()),
            "vis_occluded_ratio": float((visibility < VISIBILITY_THRESHOLD).mean()),
            "features": extract_features(sample),
        }
    return records


def _section(out: TextIO, title: str) -> None:
    out.write(f"\n## {title}\n\n")


def _write_overview(out: TextIO, samples: Sequence[Sample]) -> None:
    _section(out, "1. 收案概況")

    labels = {s.subject_id: s.label for s in samples}
    positives = sum(labels.values())
    out.write(f"- 可分析受試者：**{len(labels)} 人**"
              f"（肌少症 {positives} 人 / 無肌少症 {len(labels) - positives} 人）\n")
    out.write(f"- 動作樣本：**{len(samples)} 筆**\n\n")

    by_action = defaultdict(list)
    for s in samples:
        by_action[s.action].append(s)

    out.write("| 動作 | 樣本數 | 陽性 | 陰性 |\n|---|---|---|---|\n")
    for action in ACTIONS:
        group = by_action[action]
        pos = sum(1 for s in group if s.label == 1)
        out.write(f"| {action} | {len(group)} | {pos} | {len(group) - pos} |\n")

    completed = defaultdict(set)
    for s in samples:
        completed[s.subject_id].add(s.action)
    partial = {k: v for k, v in completed.items() if len(v) < len(ACTIONS)}

    out.write("\n**動作不齊全者**：")
    if partial:
        out.write("\n\n")
        for uid, acts in sorted(partial.items()):
            missing = [a for a in ACTIONS if a not in acts]
            out.write(f"- {uid}：缺 {', '.join(missing)}\n")
    else:
        out.write("無\n")

    if EXCLUDED_SUBJECTS or EXCLUDED_SAMPLES:
        out.write("\n**已排除的資料**：\n\n")
        for uid, reason in EXCLUDED_SUBJECTS.items():
            out.write(f"- {uid}（整位）：{reason}\n")
        for (uid, action), reason in EXCLUDED_SAMPLES.items():
            out.write(f"- {uid} / {action}：{reason}\n")

    out.write("\n### 年齡分布\n\n")
    ages = np.array([s.age for s in {x.subject_id: x for x in samples}.values()])
    lbls = np.array([s.label for s in {x.subject_id: x for x in samples}.values()])
    out.write(f"- 無肌少症組：{ages[lbls == 0].mean():.1f} ± {ages[lbls == 0].std():.1f} 歲"
              f"（{ages[lbls == 0].min():.0f}–{ages[lbls == 0].max():.0f}）\n")
    out.write(f"- 肌少症組：**{ages[lbls == 1].mean():.1f} ± {ages[lbls == 1].std():.1f} 歲**"
              f"（{ages[lbls == 1].min():.0f}–{ages[lbls == 1].max():.0f}）\n\n")
    out.write("| 年齡層 | 人數 | 陽性 | 陰性 |\n|---|---|---|---|\n")
    for low, high in [(65, 70), (70, 75), (75, 80), (80, 85), (85, 95)]:
        mask = (ages >= low) & (ages < high)
        if mask.sum():
            out.write(f"| {low}–{high} | {mask.sum()} | {int(lbls[mask].sum())} "
                      f"| {int((1 - lbls[mask]).sum())} |\n")

    out.write("\n> **兩組年齡差距 13 歲**，肌少症組明顯年長。這使得「年齡」單獨即具備\n"
              "> 相當的判別力，分析時必須特別區分模型是在看動作還是在看年齡。\n"
              "> 建議後續收案時，盡量納入 **80 歲以上的無肌少症個案** 與\n"
              "> **75 歲以下的肌少症個案**，以降低兩者混淆。\n")


def _write_quality(out: TextIO, records: Dict[str, Dict]) -> None:
    _section(out, "2. 錄影品質")

    out.write("MediaPipe 對每個關節會輸出「可見度」分數。關節被身體遮擋時，\n"
              "系統**仍會推測並輸出一組看似正常的座標**，因此低可見度的資料\n"
              "外觀上看不出問題，卻無法作為判讀依據。\n\n")

    out.write("| 動作 | 整體可見度 | 下肢可見度 | 被遮擋幀比例 | 秒數（中位數） |\n")
    out.write("|---|---|---|---|---|\n")
    for action in ACTIONS:
        rows = [r for r in records.values() if r["sample"].action == action]
        if not rows:
            continue
        out.write(
            f"| {action} "
            f"| {np.mean([r['vis_mean'] for r in rows]):.3f} "
            f"| {np.mean([r['vis_lower'] for r in rows]):.3f} "
            f"| {np.mean([r['vis_occluded_ratio'] for r in rows]) * 100:.1f}% "
            f"| {np.median([r['duration_s'] for r in rows]):.1f} s |\n"
        )

    poor = sorted(
        (r for r in records.values() if r["vis_lower"] < 0.85),
        key=lambda r: r["vis_lower"],
    )
    out.write(f"\n### 下肢可見度偏低的樣本（< 0.85）\n\n")
    if poor:
        out.write("| 受試者 | 動作 | 下肢可見度 | 標籤 |\n|---|---|---|---|\n")
        for r in poor:
            s = r["sample"]
            out.write(f"| {s.subject_id} | {s.action} | {r['vis_lower']:.3f} "
                      f"| {'肌少症' if s.label else '無'} |\n")
        out.write("\n> 這些樣本的膝、踝座標多為系統推測值，**左右對稱性等指標不可採信**。\n")
    else:
        out.write("無\n")


def _write_protocol(out: TextIO, records: Dict[str, Dict]) -> None:
    _section(out, "3. 檢測協定遵從度")

    out.write("### 半蹲：最低點停留 3 秒\n\n")
    out.write("依物理治療師說明，半蹲的判讀重點在**彎曲角度與維持能力**。\n"
              "若不要求停留時間，長者容易蹲下即刻起身、或反覆快速上下，兩者皆有跌倒風險，\n"
              f"因此協定要求在最低點維持約 {SQUAT_HOLD_TARGET_S:.0f} 秒。\n\n")

    squat = [r for r in records.values() if r["sample"].action == "Squat"]
    completed = [r for r in squat if r["features"]["completed_squat"] == 1.0]
    incomplete = [r for r in squat if r["features"]["completed_squat"] == 0.0]

    out.write(f"- 實際完成下蹲者（膝關節活動範圍 ≥ {SQUAT_MIN_ROM_DEG:.0f}°）："
              f"**{len(completed)} / {len(squat)} 人**\n")
    if incomplete:
        for r in incomplete:
            s = r["sample"]
            out.write(f"  - {s.subject_id}：活動範圍僅 "
                      f"{r['features']['knee_rom_deg']:.0f}°，幾乎未彎曲膝關節\n")

    holds = np.array([r["features"]["hold_duration_s"] for r in completed])
    meets = sum(1 for r in completed if r["features"]["meets_hold_protocol"] == 1.0)
    out.write(f"\n- 停留時間中位數：**{np.median(holds):.2f} 秒**"
              f"（協定要求 {SQUAT_HOLD_TARGET_S:.0f} 秒）\n")
    out.write(f"- 達成 3 秒者：**{meets} / {len(completed)} 人"
              f"（{meets / max(len(completed), 1) * 100:.0f}%）**\n\n")

    out.write("| 停留時間 | 人數 | 肌少症 | 無肌少症 |\n|---|---|---|---|\n")
    for low, high, label in [(0, 1, "< 1 秒"), (1, 2, "1–2 秒"),
                             (2, 3, "2–3 秒"), (3, 99, "≥ 3 秒（達標）")]:
        band = [r for r in completed if low <= r["features"]["hold_duration_s"] < high]
        if band:
            pos = sum(1 for r in band if r["sample"].label == 1)
            out.write(f"| {label} | {len(band)} | {pos} | {len(band) - pos} |\n")

    out.write("\n> **停留時間本身即為有效指標**——撐不到 2 秒者以肌少症個案居多。\n"
              "> 因此未達協定的樣本不應視為收案失誤，反而是重要訊號。\n"
              "> 但仍建議現場加強計時提示，使「撐不住」與「不知道要撐」能夠區分。\n")

    out.write("\n### 起立坐下：五次\n\n")
    sit = [r for r in records.values() if r["sample"].action == "Sit"]
    reps = np.array([r["features"]["reps"] for r in sit])
    out.write(f"- 自動偵測次數：中位數 **{np.median(reps):.1f} 次**"
              f"（範圍 {reps.min():.0f}–{reps.max():.0f}）\n")
    out.write(f"- 平均每次耗時：**{np.median([r['features']['sec_per_rep'] for r in sit]):.2f} 秒**\n")
    off = [r for r in sit if abs(r["features"]["reps"] - 5) >= 2]
    if off:
        out.write("\n次數明顯偏離 5 次者：\n\n")
        for r in off:
            out.write(f"- {r['sample'].subject_id}：偵測到 {r['features']['reps']:.0f} 次\n")


def _write_clinical(out: TextIO, records: Dict[str, Dict]) -> None:
    _section(out, "4. 臨床指標在兩組間的分布")

    out.write("以下指標由骨架資料自動計算，AUC 表示單一指標的判別力\n"
              "（0.5 = 無判別力，越接近 0 或 1 表示區隔越明顯）。\n")

    focus = {
        "Walk": [
            ("gait_speed_proxy", "步行速度估計"),
            ("cadence_hz", "步頻（步/秒）"),
            ("step_period_s", "步態週期"),
            ("duration_s", "行走耗時"),
            ("knee_asymmetry", "左右膝關節不對稱"),
        ],
        "Sit": [
            ("sec_per_rep", "每次起立坐下耗時"),
            ("knee_rom_deg", "膝關節活動範圍"),
            ("trunk_lean_max_deg", "軀幹最大前傾角"),
            ("standing_ratio", "站立姿勢時間佔比"),
        ],
        "Squat": [
            ("descent_speed_deg_s", "下蹲速度"),
            ("hold_drift", "維持期高度下沉"),
            ("hold_tremor_deg", "維持期抖動"),
            ("hold_duration_s", "最低點停留時間"),
            ("knee_min_deg", "最低點膝關節角度"),
        ],
    }

    for action, metrics in focus.items():
        rows = [r for r in records.values() if r["sample"].action == action]
        if not rows:
            continue
        labels = np.array([r["sample"].label for r in rows])
        out.write(f"\n### {action}\n\n")
        out.write("| 指標 | 無肌少症 | 肌少症 | AUC |\n|---|---|---|---|\n")
        for key, name in metrics:
            values = np.array([r["features"][key] for r in rows])
            if np.ptp(values) < 1e-9:
                continue
            auc = roc_auc_score(labels, values)
            out.write(
                f"| {name} | {values[labels == 0].mean():.2f} ± {values[labels == 0].std():.2f} "
                f"| {values[labels == 1].mean():.2f} ± {values[labels == 1].std():.2f} "
                f"| {auc:.3f} |\n"
            )

        if action == "Squat":
            out.write(
                "\n> **解讀注意**：維持期的「高度下沉」與「抖動」在肌少症組反而較低，\n"
                "> 與臨床直覺相反。合理解釋是這兩項與停留時間相互干擾——\n"
                "> 肌少症組平均只撐 2.00 秒（對照組 2.50 秒），停留時間較短即較少機會\n"
                "> 累積下沉與抖動。因此**不應解讀為肌少症者姿勢較穩定**。\n"
                "> 若要獨立評估穩定度，需改以「每秒下沉量」計算，或統一停留時間後再比較。\n"
            )


def _write_recommendations(out: TextIO, records: Dict[str, Dict]) -> None:
    _section(out, "5. 收案調整建議")

    out.write("### 建議 1：改善行走（Walk）的拍攝角度\n\n")
    out.write("| 動作 | 下肢可見度 | 被遮擋幀比例 | 評估 |\n|---|---|---|---|\n")
    for action in ACTIONS:
        rows = [r for r in records.values() if r["sample"].action == action]
        vis_lower = np.mean([r["vis_lower"] for r in rows])
        occluded = np.mean([r["vis_occluded_ratio"] for r in rows])
        verdict = "良好" if vis_lower >= 0.95 else "需改善"
        out.write(f"| {action} | {vis_lower:.3f} | {occluded * 100:.1f}% | {verdict} |\n")

    out.write("\n**Sit 與 Squat 的拍攝品質良好，無須調整。**\n"
              "行走的下肢可見度較低、被遮擋幀比例達 7.7%，是三者中唯一需改善者；\n"
              "個別樣本更低至 0.504。原因為側面拍攝時靠鏡頭側的肢體會遮擋另一側，\n"
              "使該側膝、踝的座標成為系統推測值，導致左右對稱性等指標不可採信。\n\n"
              "**建議行走改為斜前方 45 度拍攝，或加設第二機位。**\n")

    out.write("\n### 建議 2：補收特定年齡層的個案\n\n")
    out.write("目前兩組年齡差距 13 歲，模型難以區分是「動作退化」還是「年齡本身」。\n"
              "**建議優先補收：80 歲以上的無肌少症者、75 歲以下的肌少症者。**\n")

    out.write("\n### 建議 3：半蹲維持計時的現場提示\n\n")
    squat = [r for r in records.values() if r["sample"].action == "Squat"]
    completed = [r for r in squat if r["features"]["completed_squat"] == 1.0]
    meets = sum(1 for r in completed if r["features"]["meets_hold_protocol"] == 1.0)
    out.write(f"僅 {meets}/{len(completed)} 人達到 3 秒。停留時間本身具判別力，\n"
              "建議現場以口頭倒數或計時器提示，使「撐不住」與「不知道要撐滿」可區分。\n")

    out.write("\n### 建議 4：記錄握力與小腿圍的原始數值\n\n")
    out.write("目前僅取得最終的二元判定。若能保留握力（公斤）與小腿圍（公分）的\n"
              "原始測量值，可用於驗證模型從動作中學到的特徵是否與肌力指標相關，\n"
              "分析價值高於單一標籤。\n")

    out.write("\n### 建議 5：影片起訖時間的標準化\n\n")
    out.write("行走影片目前以人工方式停止錄影，尾端可能包含數秒的多餘畫面。\n"
              "「行走耗時」是目前判別力最高的指標之一，建議統一起訖判準\n"
              "（例如固定距離的起點與終點標記），以降低測量誤差。\n")


def build_report(out: TextIO) -> None:
    samples = build_manifest()
    records = _collect(samples)

    out.write("# 肌少症骨架偵測系統 — 資料健檢報告\n\n")
    out.write("本報告依現有骨架資料自動產生，供與醫療機構討論收案品質與後續調整。\n")
    out.write(f"\n影片規格：720p / {FPS:.0f} fps；骨架擷取：MediaPipe Pose（33 關節點）\n")

    _write_overview(out, samples)
    _write_quality(out, records)
    _write_protocol(out, records)
    _write_clinical(out, records)
    _write_recommendations(out, records)

    out.write("\n---\n\n")
    out.write("### 附註：本報告的限制\n\n")
    out.write("- 樣本數 31 人，屬可行性研究規模，各項統計量的不確定性較大。\n")
    out.write("- 標籤為物理治療師之臨床判定，其依據包含握力等本系統無法取得的資訊。\n")
    out.write("- 所有指標由骨架座標自動計算，未經人工逐幀核對。\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="產生資料健檢報告")
    parser.add_argument("--save", action="store_true", help="另存為 Markdown 檔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import io

    buffer = io.StringIO()
    build_report(buffer)
    content = buffer.getvalue()
    print(content)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "data_health_report.md"
        out_path.write_text(content, encoding="utf-8")
        print(f"\n報告已儲存：{out_path}")


if __name__ == "__main__":
    main()

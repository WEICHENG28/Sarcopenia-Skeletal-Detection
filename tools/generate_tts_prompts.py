"""Generate the static Traditional Chinese prompts used by the Kebbi web UI."""

from __future__ import annotations

import asyncio
import argparse
from pathlib import Path

import edge_tts


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "static" / "tts"
VOICE = "zh-TW-HsiaoChenNeural"
PROMPTS = {
    "manual_name.mp3": "您好，請在畫面輸入您的姓名，完成後按確認姓名。",
    "ask_name.mp3": "您好，請在嗶聲後，只說您的姓名，例如王小明。",
    "profile_loaded.mp3": "您好，基本資料已載入。接下來將依序進行三項動作測試。",
    "sit_prompt.mp3": "第一項是五次起立坐下。請坐在穩固且沒有輪子的椅子上，雙手交叉於胸前。聽到開始後，連續完成五次。",
    "walk_prompt.mp3": "第二項是行走測試。請站在起點，聽到開始後走向地面標示的終點。系統判斷到達終點後會自動停止。",
    "squat_prompt.mp3": "第三項是半蹲測試。請站穩，聽到開始後做一次半蹲，保持姿勢，系統錄影四秒後會自動停止。若感到不適，請立即停止。",
    "sit_ready.mp3": "請先坐穩，雙腳踩地。系統確認坐姿後會自動倒數。",
    "walk_ready.mp3": "請站在起點保持站穩。系統確認站姿後會自動倒數。",
    "squat_ready.mp3": "請站直並保持身體穩定。系統確認站姿後會自動倒數。",
    "pose_ready.mp3": "姿勢已就緒，準備倒數。",
    "count_1.mp3": "一",
    "count_2.mp3": "二",
    "count_3.mp3": "三",
    "count_4.mp3": "四",
    "count_5.mp3": "五",
    "start.mp3": "開始",
    "end.mp3": "結束",
    "enter_frame.mp3": "請完整入鏡",
    "hold.mp3": "撐住",
    "good.mp3": "很好",
    "greeting.mp3": "您好，我是肌少症預防守護者，凱比。今天由我陪您做檢測。",
    "qr_prompt.mp3": "請給我看您的 QR Code。",
    "complete_5.mp3": "已完成五下，錄影即將結束。",
    "movement_stopped.mp3": "偵測到動作已停止，錄影即將結束。",
    "squat_recorded.mp3": "半蹲動作已記錄，錄影即將結束。",
    "walk_arrived.mp3": "已到達終點，錄影即將結束。",
    "retry.mp3": "這次沒有完成，我們再試一次。",
    "analysis.mp3": "三項動作已完成，正在進行分析，請稍候。",
}


async def generate(selected: set[str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in PROMPTS.items():
        if selected is not None and filename not in selected:
            continue
        destination = OUTPUT_DIR / filename
        speech = edge_tts.Communicate(text, VOICE, rate="-5%")
        await speech.save(str(destination))
        print(f"generated {destination.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(PROMPTS))
    args = parser.parse_args()
    asyncio.run(generate(set(args.only) if args.only else None))

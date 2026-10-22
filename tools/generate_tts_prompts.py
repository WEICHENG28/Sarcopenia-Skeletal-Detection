"""Generate the static Traditional Chinese prompts used by the Kebbi web UI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "static" / "tts"
VOICE = "zh-TW-HsiaoChenNeural"
PROMPTS = {
    "manual_name.mp3": "您好，請在畫面輸入您的姓名，完成後按確認姓名。",
    "ask_name.mp3": "您好，請清楚說出您的姓名。",
    "profile_loaded.mp3": "您好，基本資料已載入。接下來將依序進行三項動作測試。",
    "sit_prompt.mp3": "第一項是五次起立坐下。請坐在穩固且沒有輪子的椅子上，雙手交叉於胸前。聽到開始後，連續完成五次。",
    "walk_prompt.mp3": "第二項是行走測試。請站在起點，聽到開始後向前行走，完成後停下。",
    "squat_prompt.mp3": "第三項是半蹲測試。請站穩，聽到開始後完成五次半蹲。若感到不適，請立即停止。",
    "count_1.mp3": "一",
    "count_2.mp3": "二",
    "count_3.mp3": "三",
    "count_4.mp3": "四",
    "count_5.mp3": "五",
    "start.mp3": "開始",
    "enter_frame.mp3": "請完整入鏡",
    "hold.mp3": "撐住",
    "good.mp3": "很好",
    "complete_5.mp3": "已完成五下，錄影即將結束。",
    "movement_stopped.mp3": "偵測到動作已停止，錄影即將結束。",
    "analysis.mp3": "三項動作已完成，正在進行分析，請稍候。",
}


async def generate() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in PROMPTS.items():
        destination = OUTPUT_DIR / filename
        speech = edge_tts.Communicate(text, VOICE, rate="-5%")
        await speech.save(str(destination))
        print(f"generated {destination.name}")


if __name__ == "__main__":
    asyncio.run(generate())

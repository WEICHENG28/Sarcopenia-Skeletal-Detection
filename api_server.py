"""肌少症篩檢伺服器：影片上傳 → 骨架擷取 → 風險評估。

整合三個部分：
- 前端網頁（``static/index.html``）
- 影片轉骨架（``extract_to_json.py``，MediaPipe world landmarks）
- 風險評估模型（``src/predict.py``，見 ``docs/API_CONTRACT.md``）

推論以**行程內呼叫**進行，不另外開子程序——舊版透過 subprocess 呼叫
另一份 Python 環境，除錯困難且需維護兩套相依套件。

啟動：
    python api_server.py
    # 或
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.dataset_manifest import ACTIONS
from src.predict import predict

# 上傳資料與訓練資料分開存放。
# 訓練集（data/A_*_json/）是經過稽核與排除規則篩選的，
# 若讓伺服器直接寫入，會在無人察覺的情況下污染實驗結果。
# 由上傳資料進入訓練集應為明確的人工策展步驟。
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VIDEOS_DIR = UPLOAD_DIR / "videos"
JSON_DIR = UPLOAD_DIR / "json"
RECORDS_CSV = UPLOAD_DIR / "records.csv"

EXTRACT_SCRIPT = BASE_DIR / "extract_to_json.py"
STATIC_DIR = BASE_DIR / "static"

ACTION_ALIASES = {a.lower(): a for a in ACTIONS}

CSV_HEADERS = [
    "user_id", "age", "height_cm", "weight_kg", "gender",
    "action", "video_path", "json_path",
    "therapist_label", "labeled_at",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_filename_component(value: str) -> str:
    """把使用者輸入轉成安全的檔名片段，避免路徑穿越。"""
    cleaned = _SAFE_NAME.sub("_", str(value).strip())
    return cleaned or "unknown"


def normalize_action(action: str) -> str | None:
    """把 sit / SIT / Sit 統一成模型使用的 "Sit"。"""
    return ACTION_ALIASES.get(str(action).strip().lower())


def init_storage() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    if not RECORDS_CSV.exists():
        pd.DataFrame(columns=CSV_HEADERS).to_csv(
            RECORDS_CSV, index=False, encoding="utf-8-sig"
        )


def find_user_action_jsons(user_id: str) -> Dict[str, str]:
    """找出該受測者已完成的動作骨架檔。"""
    found: Dict[str, str] = {}
    for action in ACTIONS:
        path = JSON_DIR / f"{user_id}_{action}.json"
        if path.exists():
            found[action] = str(path)
    return found


def summarize_for_frontend(result: Dict[str, Any]) -> list[str]:
    """把模型輸出整理成前端可直接列出的說明文字。

    刻意使用「模型主要參考」而非「問題部位」——
    本研究無部位層級的臨床標註可供驗證，不得呈現為診斷結論。
    """
    lines: list[str] = []

    regions = result.get("region_attention") or {}
    if regions:
        ranked = sorted(regions.items(), key=lambda kv: kv[1], reverse=True)[:3]
        parts = "、".join(f"{name}（{share:.0%}）" for name, share in ranked)
        lines.append(f"模型主要參考的動作部位：{parts}")

    for action, detail in (result.get("per_action") or {}).items():
        if detail.get("performed") and detail.get("probability") is not None:
            lines.append(f"{action} 動作風險指數：{detail['probability']:.0%}")

    # warnings 與 uncertainty_reasons 由前端獨立顯示，不混入影響因子
    return lines


def run_inference(user_id: str, age: float, height_cm: float,
                  weight_kg: float, gender: str) -> Dict[str, Any]:
    """呼叫模型評估風險。任何失敗都轉成結構化回應，不讓伺服器崩潰。"""
    actions = find_user_action_jsons(user_id)
    if not actions:
        return {
            "status": "skipped",
            "message": "尚無可用的動作骨架檔，略過評估",
            "risk_probability": None,
        }

    try:
        result = predict(
            profile={
                "age": age,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "gender": gender,
            },
            actions=actions,
        ).to_dict()
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "message": f"模型檔尚未就緒：{exc}。請先執行 python -m src.train_final",
            "risk_probability": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"評估失敗：{exc}",
            "risk_probability": None,
        }

    result["status"] = "success" if result.get("ok") else "error"
    result["top_features"] = summarize_for_frontend(result)
    return result


app = FastAPI(title="肌少症篩檢 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup_event() -> None:
    init_storage()


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康檢查：供部署時確認模型檔是否就緒。"""
    models_dir = BASE_DIR / "models"
    required = [
        "metadata.json", "stacking.joblib",
        "stgcn_sit.pth", "stgcn_walk.pth", "stgcn_squat.pth",
    ]
    missing = [f for f in required if not (models_dir / f).exists()]
    return {
        "status": "ok" if not missing else "models_missing",
        "actions": list(ACTIONS),
        "missing_files": missing,
        "hint": "執行 python -m src.train_final 產生模型檔" if missing else "",
    }


@app.post("/upload_data/")
async def upload_data(
    user_id: str = Form(...),
    age: float = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    gender: str = Form(...),
    action: str = Form(...),
    video_file: UploadFile = File(...),
):
    """接收影片，擷取骨架後評估風險。

    流程：存檔 → MediaPipe 轉 JSON → 寫入紀錄 → 以目前已完成的動作評估。
    每次上傳都會用**該受測者所有已完成的動作**重新評估，
    因此三段影片分次上傳時，分數會隨資料增加而更新。
    """
    init_storage()

    normalized = normalize_action(action)
    if normalized is None:
        video_file.file.close()
        return {
            "status": "skipped",
            "message": f"不支援的動作「{action}」",
            "accepted_actions": list(ACTIONS),
        }

    safe_user = sanitize_filename_component(user_id)
    base_name = f"{safe_user}_{normalized}"
    video_path = VIDEOS_DIR / f"{base_name}.mp4"
    final_json = JSON_DIR / f"{base_name}.json"

    # 1) 存檔
    try:
        with video_path.open("wb") as f:
            shutil.copyfileobj(video_file.file, f)
    except Exception as exc:
        raise HTTPException(500, f"影片存檔失敗：{exc}") from exc
    finally:
        video_file.file.close()

    # 2) 擷取骨架
    if not EXTRACT_SCRIPT.exists():
        raise HTTPException(500, f"找不到轉換腳本：{EXTRACT_SCRIPT}")

    try:
        subprocess.run(
            [sys.executable, str(EXTRACT_SCRIPT),
             "-i", str(video_path), "--dataset_dir", str(JSON_DIR)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(500, f"MediaPipe 轉換失敗：{detail}") from exc

    produced = JSON_DIR / f"{video_path.stem}_3d_data.json"
    if not produced.exists():
        raise HTTPException(500, f"轉換後找不到骨架檔：{produced}")
    final_json.unlink(missing_ok=True)
    shutil.move(str(produced), str(final_json))

    # 3) 寫入紀錄
    try:
        pd.DataFrame([{
            "user_id": safe_user,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender,
            "action": normalized,
            "video_path": str(video_path.relative_to(BASE_DIR)),
            "json_path": str(final_json.relative_to(BASE_DIR)),
            "therapist_label": "",
            "labeled_at": "",
        }]).to_csv(
            RECORDS_CSV, mode="a",
            header=not RECORDS_CSV.exists() or RECORDS_CSV.stat().st_size == 0,
            index=False, encoding="utf-8-sig",
        )
    except Exception as exc:
        raise HTTPException(500, f"紀錄寫入失敗：{exc}") from exc

    # 4) 評估
    inference_result = run_inference(safe_user, age, height_cm, weight_kg, gender)

    return {
        "status": "success",
        "message": "上傳與分析完成",
        "user_id": safe_user,
        "action": normalized,
        "video_path": str(video_path.relative_to(BASE_DIR)),
        "json_path": str(final_json.relative_to(BASE_DIR)),
        "completed_actions": sorted(find_user_action_jsons(safe_user)),
        "inference_result": inference_result,
    }


@app.get("/result/{user_id}")
def get_result(user_id: str, age: float, height_cm: float,
               weight_kg: float, gender: str = "") -> Dict[str, Any]:
    """以現有骨架檔重新評估，不需重新上傳影片。"""
    safe_user = sanitize_filename_component(user_id)
    return {
        "user_id": safe_user,
        "completed_actions": sorted(find_user_action_jsons(safe_user)),
        "inference_result": run_inference(
            safe_user, age, height_cm, weight_kg, gender
        ),
    }


@app.post("/label/")
async def save_label(user_id: str = Form(...), label: int = Form(...)):
    """治療師標記診斷結果，寫回該受測者的所有紀錄。"""
    if label not in (0, 1):
        raise HTTPException(400, "label 必須為 0 或 1")

    safe_user = sanitize_filename_component(user_id)
    labeled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not RECORDS_CSV.exists() or RECORDS_CSV.stat().st_size == 0:
        raise HTTPException(404, "尚無資料，請先上傳影片")

    try:
        frame = pd.read_csv(RECORDS_CSV, encoding="utf-8-sig")
        for column in ("therapist_label", "labeled_at"):
            if column not in frame.columns:
                frame[column] = ""
            frame[column] = frame[column].astype(object)

        mask = frame["user_id"].astype(str) == safe_user
        if not mask.any():
            raise HTTPException(404, f"找不到受測者：{safe_user}")

        frame.loc[mask, "therapist_label"] = str(label)
        frame.loc[mask, "labeled_at"] = labeled_at
        frame.to_csv(RECORDS_CSV, index=False, encoding="utf-8-sig")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"標記儲存失敗：{exc}") from exc

    return {
        "status": "success",
        "user_id": safe_user,
        "label": label,
        "labeled_at": labeled_at,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

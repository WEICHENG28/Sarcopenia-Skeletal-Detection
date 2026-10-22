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
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal
from xml.etree import ElementTree

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.dataset_manifest import ACTIONS
from src.predict import predict
from speech_recognition_service import OfflineSpeechRecognizer, SpeechRecognitionError
from webcam_capture import CAMERA_NAME, CameraHub, find_ffmpeg

# 上傳資料與訓練資料分開存放。
# 訓練集（data/A_*_json/）是經過稽核與排除規則篩選的，
# 若讓伺服器直接寫入，會在無人察覺的情況下污染實驗結果。
# 由上傳資料進入訓練集應為明確的人工策展步驟。
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VIDEOS_DIR = UPLOAD_DIR / "videos"
JSON_DIR = UPLOAD_DIR / "json"
RECORDS_CSV = UPLOAD_DIR / "records.csv"
CAMERA_RECORDINGS_DIR = UPLOAD_DIR / "camera_recordings"
ROSTER_PATH = BASE_DIR / "outputs" / "automation-roster" / "受測者名冊.xlsx"

EXTRACT_SCRIPT = BASE_DIR / "extract_to_json.py"
STATIC_DIR = BASE_DIR / "static"

ACTION_ALIASES = {a.lower(): a for a in ACTIONS}

CSV_HEADERS = [
    "user_id", "age", "height_cm", "weight_kg", "gender",
    "action", "video_path", "json_path",
    "therapist_label", "labeled_at",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_NAME_NOISE = re.compile(r"[\s,，.。．·・\-_/\\]+")
_ROSTER_HEADERS = {
    "participant_id", "name", "name_aliases", "age",
    "height_cm", "weight_kg", "gender", "enabled",
}


class NameMatchRequest(BaseModel):
    """瀏覽器語音辨識可能回傳多個候選字串。"""

    transcripts: list[str] = Field(min_length=1, max_length=5)


class SpeechListenRequest(BaseModel):
    """Request one short utterance from the PC-connected C922 microphone."""

    purpose: Literal["name", "confirmation"] = "name"
    duration_seconds: float = Field(default=4.5, ge=2.0, le=8.0)


class ParticipantSelection(BaseModel):
    participant_id: str


class AutomationFailure(BaseModel):
    message: str = Field(min_length=1, max_length=200)


_automation_lock = threading.Lock()
_automation_state: Dict[str, Any] = {
    "revision": 0,
    "state": "idle",
    "message": "尚未啟動",
    "participant": None,
    "updated_at": datetime.now().isoformat(timespec="seconds"),
}


def _set_automation_state(state: str, message: str,
                          participant: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with _automation_lock:
        _automation_state["revision"] += 1
        _automation_state["state"] = state
        _automation_state["message"] = message
        _automation_state["participant"] = participant
        _automation_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return dict(_automation_state)


def get_automation_state() -> Dict[str, Any]:
    with _automation_lock:
        return dict(_automation_state)


def normalize_spoken_name(value: str) -> str:
    """只移除不影響姓名的空白與標點，不做可能配錯人的模糊比對。"""
    return _NAME_NOISE.sub("", str(value or "")).casefold()


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str],
                     namespace: str) -> Any:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        inline = cell.find(f"{namespace}is")
        return "" if inline is None else "".join(inline.itertext())
    value_node = cell.find(f"{namespace}v")
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type == "b":
        return raw == "1"
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


def read_roster_rows(path: Path = ROSTER_PATH) -> list[Dict[str, Any]]:
    """以標準函式庫讀取名冊，不要求伺服器額外安裝 Excel 套件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到名冊：{path}")

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in shared_root.findall(f"{namespace}si")]
        sheet_root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    parsed_rows: list[Dict[int, Any]] = []
    for row in sheet_root.findall(f".//{namespace}row"):
        values: Dict[int, Any] = {}
        for cell in row.findall(f"{namespace}c"):
            reference = cell.attrib.get("r", "")
            column_letters = re.match(r"[A-Z]+", reference)
            if not column_letters:
                continue
            index = 0
            for letter in column_letters.group(0):
                index = index * 26 + ord(letter) - 64
            values[index - 1] = _xlsx_cell_value(cell, shared, namespace)
        parsed_rows.append(values)

    if not parsed_rows:
        return []
    headers = {index: str(value).strip() for index, value in parsed_rows[0].items()}
    missing = _ROSTER_HEADERS - set(headers.values())
    if missing:
        raise ValueError(f"名冊缺少欄位：{', '.join(sorted(missing))}")

    records: list[Dict[str, Any]] = []
    for raw_row in parsed_rows[1:]:
        record = {header: raw_row.get(index, "") for index, header in headers.items()}
        participant_id = str(record.get("participant_id", "")).strip()
        name = str(record.get("name", "")).strip()
        enabled = str(record.get("enabled", "TRUE")).strip().lower()
        if not participant_id or not name or enabled in {"false", "0", "否", "no"}:
            continue
        try:
            profile = {
                "participant_id": participant_id,
                "name": name,
                "name_aliases": str(record.get("name_aliases", "")).strip(),
                "age": int(float(record["age"])),
                "height_cm": float(record["height_cm"]),
                "weight_kg": float(record["weight_kg"]),
                "gender": str(record["gender"]).strip().lower(),
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"受測者 {participant_id} 的基本資料格式錯誤") from exc
        if profile["gender"] not in {"male", "female"}:
            raise ValueError(f"受測者 {participant_id} 的 gender 必須為 male 或 female")
        records.append(profile)
    return records


def public_profile(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: record[key] for key in (
        "participant_id", "name", "age", "height_cm", "weight_kg", "gender"
    )}


def match_roster_names(records: list[Dict[str, Any]],
                       transcripts: list[str]) -> list[Dict[str, Any]]:
    """依姓名或名冊別名的精確字串配對，不使用相似音猜測。

    語音模型常回傳「我的名字是王小明」而不是只有「王小明」，因此除了
    整句相等，也允許姓名／別名完整出現在句子中。若一句話同時命中多人，
    呼叫端仍會要求本人確認，不會自行挑選。
    """
    heard = {normalize_spoken_name(text) for text in transcripts if text.strip()}
    matches: list[Dict[str, Any]] = []
    for record in records:
        accepted = {normalize_spoken_name(record["name"])}
        accepted.update(
            normalize_spoken_name(alias)
            for alias in re.split(r"[;；]", record.get("name_aliases", ""))
            if alias.strip()
        )
        if any(
            spoken == name or name in spoken
            for spoken in heard
            for name in accepted
            if name
        ):
            matches.append(public_profile(record))
    return matches


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

    lines.extend(result.get("warnings") or [])
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
camera = CameraHub(CAMERA_RECORDINGS_DIR)
speech_recognizer = OfflineSpeechRecognizer(find_ffmpeg())

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
    camera.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    camera.stop()


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/control")
def serve_control():
    """電腦端只需開啟此頁啟動／中止一輪自動檢測。"""
    return FileResponse(str(STATIC_DIR / "control.html"))


@app.get("/participants/status")
def participant_roster_status() -> Dict[str, Any]:
    try:
        records = read_roster_rows()
        ids = [item["participant_id"] for item in records]
        duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
        return {
            "ready": bool(records) and not duplicate_ids,
            "count": len(records),
            "duplicate_ids": duplicate_ids,
            "message": "名冊可使用" if records and not duplicate_ids else "名冊沒有可用資料或有重複編號",
        }
    except Exception as exc:
        return {"ready": False, "count": 0, "duplicate_ids": [], "message": str(exc)}


@app.post("/participants/match")
def match_participant(request: NameMatchRequest) -> Dict[str, Any]:
    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc
    matches = match_roster_names(records, request.transcripts)
    return {
        "match_count": len(matches),
        "candidates": matches,
        "requires_confirmation": len(matches) == 1,
    }


@app.get("/speech/status")
def speech_status() -> Dict[str, Any]:
    """Report local ASR availability without loading the large model."""
    return speech_recognizer.status()


@app.post("/speech/listen")
def listen_for_speech(request: SpeechListenRequest) -> Dict[str, Any]:
    """Capture a short C922 utterance and transcribe it locally."""
    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc

    roster_hints: list[str] = []
    if request.purpose == "name":
        for record in records:
            roster_hints.append(record["name"])
            roster_hints.extend(
                alias.strip()
                for alias in re.split(r"[;；]", record.get("name_aliases", ""))
                if alias.strip()
            )
    try:
        result = speech_recognizer.listen(
            purpose=request.purpose,
            roster_hints=roster_hints,
            duration_seconds=request.duration_seconds,
        )
    except SpeechRecognitionError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"status": "recognized", **result}


@app.get("/automation/state")
def automation_state() -> Dict[str, Any]:
    return get_automation_state()


@app.post("/automation/start")
def start_automation() -> Dict[str, Any]:
    roster = participant_roster_status()
    if not roster["ready"]:
        raise HTTPException(409, f"無法啟動：{roster['message']}")
    if camera.get_frame(timeout=0.5) is None:
        raise HTTPException(409, "無法啟動：C922 尚未產生畫面")
    return _set_automation_state("armed", "已啟動，等待受測者完整入鏡")


@app.post("/automation/stop")
def stop_automation() -> Dict[str, Any]:
    if camera.is_recording:
        try:
            camera.stop_recording()
        except RuntimeError:
            pass
    return _set_automation_state("idle", "已由電腦端停止")


@app.post("/automation/person-detected")
def automation_person_detected() -> Dict[str, Any]:
    current = get_automation_state()
    if current["state"] != "armed":
        return current
    return _set_automation_state("identifying", "已偵測到受測者，正在詢問姓名")


@app.post("/automation/profile-loaded")
def automation_profile_loaded(selection: ParticipantSelection) -> Dict[str, Any]:
    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc
    record = next(
        (item for item in records if item["participant_id"] == selection.participant_id),
        None,
    )
    if record is None:
        raise HTTPException(404, "名冊中找不到此受測者")
    profile = public_profile(record)
    return _set_automation_state(
        "profile_loaded", f"已確認 {profile['name']}，準備開始動作測試", profile
    )


@app.post("/automation/testing")
def automation_testing(selection: ParticipantSelection) -> Dict[str, Any]:
    current = get_automation_state()
    if not current.get("participant") or current["participant"]["participant_id"] != selection.participant_id:
        raise HTTPException(409, "尚未完成受測者確認")
    return _set_automation_state("testing", "自動動作測試進行中", current["participant"])


@app.post("/automation/completed")
def automation_completed(selection: ParticipantSelection) -> Dict[str, Any]:
    current = get_automation_state()
    return _set_automation_state("completed", "三項錄影已完成，正在分析或顯示結果", current.get("participant"))


@app.post("/automation/failed")
def automation_failed(failure: AutomationFailure) -> Dict[str, Any]:
    return _set_automation_state("error", failure.message)


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
        "camera": CAMERA_NAME,
        "camera_ready": camera.get_frame(timeout=0.1) is not None,
        "camera_recording": camera.is_recording,
        "speech": speech_recognizer.status(),
        "roster": participant_roster_status(),
        "automation": get_automation_state(),
    }


@app.get("/camera/frame.jpg")
def camera_frame() -> Response:
    """Return the latest PC-connected C922 frame for the Kebbi display."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(503, "C922 尚未產生畫面")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/camera/stream")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        camera.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/camera/recordings/start")
def start_camera_recording() -> Dict[str, Any]:
    try:
        recording_id = camera.start_recording()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": "recording", "recording_id": recording_id}


@app.post("/camera/recordings/stop")
def stop_camera_recording() -> Dict[str, Any]:
    try:
        result = camera.stop_recording()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    recording_id = str(result["recording_id"])
    return {
        "status": "ready",
        "recording_id": recording_id,
        "duration_seconds": result["duration_seconds"],
        "size_bytes": result["size_bytes"],
        "download_url": f"/camera/recordings/{recording_id}",
    }


@app.get("/camera/recordings/{recording_id}")
def get_camera_recording(recording_id: str) -> FileResponse:
    safe_id = sanitize_filename_component(recording_id)
    if safe_id != recording_id:
        raise HTTPException(400, "無效的錄影編號")
    path = CAMERA_RECORDINGS_DIR / f"{safe_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "找不到錄影檔")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=f"{safe_id}.mp4",
    )


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

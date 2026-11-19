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
import logging
import os
import re
import secrets
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
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.dataset_manifest import ACTIONS
from src.predict import predict
from speech_recognition_service import OfflineSpeechRecognizer, SpeechRecognitionError
from webcam_capture import CAMERA_NAME, CameraHub

logger = logging.getLogger("sarcopenia.speech")

# 上傳資料與訓練資料分開存放。
# 訓練集（data/A_*_json/）是經過稽核與排除規則篩選的，
# 若讓伺服器直接寫入，會在無人察覺的情況下污染實驗結果。
# 由上傳資料進入訓練集應為明確的人工策展步驟。
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VIDEOS_DIR = UPLOAD_DIR / "videos"
JSON_DIR = UPLOAD_DIR / "json"
RECORDS_CSV = UPLOAD_DIR / "records.csv"
PROFESSIONAL_REVIEWS_PATH = UPLOAD_DIR / "professional_reviews.json"
CAMERA_RECORDINGS_DIR = UPLOAD_DIR / "camera_recordings"
ROSTER_PATH = BASE_DIR / "outputs" / "automation-roster" / "受測者名冊.xlsx"
MAX_SPEECH_UPLOAD_BYTES = 5 * 1024 * 1024
CLINICIAN_TOKEN = os.getenv("CLINICIAN_TOKEN", "").strip()

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
_QR_PAYLOAD = re.compile(r"SARCOPENIA:([A-Za-z0-9_-]{1,64})")
EXHIBITION_TTS_FILES = (
    "manual_name.mp3", "ask_name.mp3", "profile_loaded.mp3",
    "sit_prompt.mp3", "walk_prompt.mp3", "squat_prompt.mp3",
    "sit_ready.mp3", "walk_ready.mp3", "squat_ready.mp3", "pose_ready.mp3",
    "count_1.mp3", "count_2.mp3", "count_3.mp3", "count_4.mp3", "count_5.mp3",
    "start.mp3", "end.mp3", "enter_frame.mp3", "hold.mp3", "good.mp3", "greeting.mp3",
    "qr_prompt.mp3", "chime.wav",
    "complete_5.mp3", "movement_stopped.mp3", "squat_recorded.mp3",
    "walk_arrived.mp3", "retry.mp3", "analysis.mp3",
)
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
    expected_state: Literal[
        "armed", "scanning_qr", "profile_loaded", "testing", "completed"
    ]
    expected_revision: int = Field(ge=0)
    participant_id: str | None = Field(default=None, max_length=64)


class AutomationStandby(BaseModel):
    """自動偵測模式下，把一輪流程安靜地交回待命狀態。"""

    message: str = Field(min_length=1, max_length=200)
    expected_state: Literal["scanning_qr", "profile_loaded", "completed", "error"]
    expected_revision: int = Field(ge=0)


class ClinicianReviewRequest(BaseModel):
    participant_id: str = Field(min_length=1, max_length=64)
    assessment: Literal[
        "insufficient",
        "screen_negative",
        "possible_sarcopenia",
        "refer_for_confirmation",
    ]
    note: str = Field(default="", max_length=500)


_automation_lock = threading.Lock()
_professional_review_lock = threading.Lock()
_automation_state: Dict[str, Any] = {
    "revision": 0,
    "state": "idle",
    "message": "尚未啟動",
    "participant": None,
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    # 自動偵測模式：有人走近並站定就自動開始；中止或結束後回到待命而非停止。
    "auto_mode": False,
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


def _set_automation_state_if_current(
    expected_state: str,
    expected_revision: int,
    state: str,
    message: str,
    participant: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Atomically transition only if no stop/restart occurred meanwhile."""
    with _automation_lock:
        if (_automation_state["state"] != expected_state
                or _automation_state["revision"] != expected_revision):
            return None
        _automation_state["revision"] += 1
        _automation_state["state"] = state
        _automation_state["message"] = message
        _automation_state["participant"] = participant
        _automation_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return dict(_automation_state)


def get_automation_state() -> Dict[str, Any]:
    with _automation_lock:
        return dict(_automation_state)


def require_clinician_token(
    x_clinician_token: str | None = Header(default=None, alias="X-Clinician-Token"),
) -> None:
    """Fail closed unless an explicit local clinician token is configured."""
    if not CLINICIAN_TOKEN:
        raise HTTPException(503, "專業判讀後台尚未設定 CLINICIAN_TOKEN")
    supplied = x_clinician_token or ""
    if not secrets.compare_digest(supplied, CLINICIAN_TOKEN):
        raise HTTPException(401, "判讀權杖無效")


def normalize_spoken_name(value: str) -> str:
    """只移除不影響姓名的空白與標點，不做可能配錯人的模糊比對。"""
    return _NAME_NOISE.sub("", str(value or "")).casefold()


def parse_qr_participant_id(payload: str) -> str | None:
    """Accept only the documented, privacy-minimal QR payload format."""
    match = _QR_PAYLOAD.fullmatch(str(payload or ""))
    return match.group(1) if match else None


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


def _load_professional_reviews() -> Dict[str, Dict[str, str]]:
    if not PROFESSIONAL_REVIEWS_PATH.exists():
        return {}
    try:
        raw = json.loads(PROFESSIONAL_REVIEWS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "專業判讀紀錄無法讀取") from exc
    if not isinstance(raw, dict):
        raise HTTPException(500, "專業判讀紀錄格式錯誤")
    return raw


def _write_professional_reviews(reviews: Dict[str, Dict[str, str]]) -> None:
    PROFESSIONAL_REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = PROFESSIONAL_REVIEWS_PATH.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(PROFESSIONAL_REVIEWS_PATH)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(500, "專業判讀紀錄無法儲存") from exc


def _clinician_record_summaries() -> list[Dict[str, Any]]:
    if not RECORDS_CSV.exists() or RECORDS_CSV.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(RECORDS_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(500, "收案紀錄無法讀取") from exc
    if "user_id" not in frame.columns or "action" not in frame.columns:
        raise HTTPException(500, "收案紀錄缺少必要欄位")

    reviews = _load_professional_reviews()
    records: list[Dict[str, Any]] = []
    for participant_id in sorted(set(frame["user_id"].astype(str))):
        if not participant_id:
            continue
        participant_rows = frame[frame["user_id"].astype(str) == participant_id]
        completed = [
            action for action in ACTIONS
            if action in set(participant_rows["action"].astype(str))
        ]
        review = reviews.get(participant_id, {})
        records.append({
            "participant_id": participant_id,
            "completed_actions": completed,
            "professional_review": review.get("assessment", ""),
            "review_note": review.get("note", ""),
            "reviewed_at": review.get("reviewed_at", ""),
        })
    return records


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
camera = CameraHub(CAMERA_RECORDINGS_DIR)
speech_recognizer = OfflineSpeechRecognizer(camera.ffmpeg)

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
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/control")
def serve_control():
    """電腦端只需開啟此頁啟動／中止一輪自動檢測。"""
    return FileResponse(
        str(STATIC_DIR / "control.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/demo")
def serve_demo():
    """Public, inert concept page without roster, camera, or result access."""
    return FileResponse(
        str(STATIC_DIR / "demo.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/clinician")
def serve_clinician():
    """Clinician UI; its data APIs separately require the configured token."""
    return FileResponse(
        str(STATIC_DIR / "clinician.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/clinician/records")
def clinician_records(
    _authorized: None = Depends(require_clinician_token),
) -> Dict[str, Any]:
    return {"records": _clinician_record_summaries()}


@app.post("/clinician/review")
def save_clinician_review(
    review: ClinicianReviewRequest,
    _authorized: None = Depends(require_clinician_token),
) -> Dict[str, Any]:
    safe_participant_id = sanitize_filename_component(review.participant_id)
    if safe_participant_id != review.participant_id:
        raise HTTPException(400, "受測者編號格式不符")
    known_ids = {
        item["participant_id"] for item in _clinician_record_summaries()
    }
    if safe_participant_id not in known_ids:
        raise HTTPException(404, "正式收案紀錄中找不到此受測者")

    reviewed_at = datetime.now().isoformat(timespec="seconds")
    with _professional_review_lock:
        reviews = _load_professional_reviews()
        reviews[safe_participant_id] = {
            "assessment": review.assessment,
            "note": review.note.strip(),
            "reviewed_at": reviewed_at,
        }
        _write_professional_reviews(reviews)
    return {
        "status": "success",
        "participant_id": safe_participant_id,
        "assessment": review.assessment,
        "reviewed_at": reviewed_at,
    }


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


def _speech_roster_hints(records: list[Dict[str, Any]], purpose: str) -> list[str]:
    hints: list[str] = []
    if purpose == "name":
        for record in records:
            hints.append(record["name"])
            hints.extend(
                alias.strip()
                for alias in re.split(r"[;；]", record.get("name_aliases", ""))
                if alias.strip()
            )
    return hints


@app.post("/speech/listen")
def listen_for_speech(request: SpeechListenRequest) -> Dict[str, Any]:
    """Capture a short C922 utterance and transcribe it locally."""
    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc

    roster_hints = _speech_roster_hints(records, request.purpose)
    try:
        result = speech_recognizer.listen(
            purpose=request.purpose,
            roster_hints=roster_hints,
            duration_seconds=request.duration_seconds,
        )
    except SpeechRecognitionError as exc:
        logger.warning("C922 speech recognition failed: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    return {"status": "recognized", **result}


@app.post("/speech/transcribe-upload")
async def transcribe_uploaded_speech(
    request: Request,
    purpose: Literal["name", "confirmation"] = "name",
    audio: UploadFile = File(...),
) -> Dict[str, Any]:
    """Transcribe a short recording uploaded from Kebbi's built-in microphone."""
    try:
        audio_bytes = await audio.read(MAX_SPEECH_UPLOAD_BYTES + 1)
    finally:
        await audio.close()
    if not audio_bytes:
        raise HTTPException(400, "凱比麥克風沒有收到音訊")
    if len(audio_bytes) > MAX_SPEECH_UPLOAD_BYTES:
        raise HTTPException(413, "凱比麥克風音訊超過 5 MB")

    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        result = await run_in_threadpool(
            speech_recognizer.transcribe_upload,
            audio_bytes,
            content_type=audio.content_type or request.headers.get("content-type", ""),
            purpose=purpose,
            roster_hints=_speech_roster_hints(records, purpose),
        )
    except SpeechRecognitionError as exc:
        logger.warning("Kebbi speech upload failed: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    return {"status": "recognized", "source": "kebbi_microphone", **result}


@app.get("/automation/state")
def automation_state() -> Dict[str, Any]:
    return get_automation_state()


def _require_ready_to_run() -> None:
    roster = participant_roster_status()
    if not roster["ready"]:
        raise HTTPException(409, f"無法啟動：{roster['message']}")
    if camera.get_frame(timeout=0.5) is None:
        raise HTTPException(409, "無法啟動：C922 尚未產生畫面")


@app.post("/automation/start")
def start_automation() -> Dict[str, Any]:
    _require_ready_to_run()
    # Full-body framing is intentionally deferred until after identity has
    # been confirmed.
    return _set_automation_state("scanning_qr", "已啟動 QR Code 掃描")


@app.post("/automation/arm")
def arm_automation() -> Dict[str, Any]:
    _require_ready_to_run()
    with _automation_lock:
        _automation_state["auto_mode"] = True
    return _set_automation_state("armed", "自動偵測待命中：有人走近並站定後會自動開始")


@app.post("/automation/disarm")
def disarm_automation() -> Dict[str, Any]:
    with _automation_lock:
        _automation_state["auto_mode"] = False
    current = get_automation_state()
    if current["state"] != "armed":
        # 不打斷進行中的一輪，只是結束後不再自動回到待命。
        return current
    return _set_automation_state("idle", "已關閉自動偵測")


@app.post("/automation/stop")
def stop_automation() -> Dict[str, Any]:
    if camera.is_recording:
        try:
            camera.stop_recording()
        except RuntimeError:
            pass
    if get_automation_state()["auto_mode"]:
        return _set_automation_state("armed", "已中止本輪，回到自動偵測待命")
    return _set_automation_state("idle", "已由電腦端停止")


@app.post("/automation/standby")
def automation_standby(request: AutomationStandby) -> Dict[str, Any]:
    current = get_automation_state()
    if not current["auto_mode"]:
        raise HTTPException(409, "目前不是自動偵測模式")
    state = _set_automation_state_if_current(
        request.expected_state,
        request.expected_revision,
        "armed",
        request.message,
    )
    if state is None:
        raise HTTPException(409, "自動流程狀態已改變，忽略回到待命")
    return state


@app.post("/automation/person-detected")
def automation_person_detected() -> Dict[str, Any]:
    current = get_automation_state()
    if current["state"] != "armed":
        return current
    state = _set_automation_state_if_current(
        "armed",
        current["revision"],
        "scanning_qr",
        "已偵測到受測者，請將 QR Code 對準 C922",
    )
    return state or get_automation_state()


@app.post("/automation/qr-scan")
def automation_qr_scan() -> Dict[str, Any]:
    """Decode the latest C922 frame and exactly match one roster identifier."""
    current = get_automation_state()
    if current["state"] != "scanning_qr":
        raise HTTPException(409, "目前不是 QR Code 身分確認階段")

    try:
        payloads = camera.decode_qr_codes(timeout=0.5)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    if not payloads:
        return {"status": "waiting", "message": "尚未辨識到 QR Code"}

    participant_ids = [
        participant_id
        for payload in payloads
        if (participant_id := parse_qr_participant_id(payload)) is not None
    ]
    if not participant_ids:
        return {
            "status": "invalid",
            "message": "QR Code 格式不符；請使用本系統核發的受測者編號 QR Code",
        }
    unique_ids = list(dict.fromkeys(participant_ids))
    if len(unique_ids) != 1:
        return {"status": "invalid", "message": "畫面中有多個不同的受測者 QR Code"}

    try:
        records = read_roster_rows()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(503, str(exc)) from exc
    matches = [record for record in records if record["participant_id"] == unique_ids[0]]
    if len(matches) != 1:
        return {"status": "invalid", "message": "QR Code 編號未在名冊中找到唯一對應資料"}

    profile = public_profile(matches[0])
    state = _set_automation_state_if_current(
        "scanning_qr",
        current["revision"],
        "profile_loaded",
        f"QR Code 已確認 {profile['name']}，請退後並讓全身完整入鏡",
        profile,
    )
    if state is None:
        raise HTTPException(409, "自動流程已停止或重新啟動，忽略本次 QR 掃描結果")
    return {
        "status": "matched",
        "participant": profile,
        "automation": state,
    }


@app.post("/automation/profile-loaded")
def automation_profile_loaded(selection: ParticipantSelection) -> Dict[str, Any]:
    current = get_automation_state()
    if current["state"] != "scanning_qr":
        raise HTTPException(409, "目前不是身分確認階段")
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
    state = _set_automation_state_if_current(
        "scanning_qr",
        current["revision"],
        "profile_loaded",
        f"已由姓名備援確認 {profile['name']}，請退後並讓全身完整入鏡",
        profile,
    )
    if state is None:
        raise HTTPException(409, "自動流程已停止或重新啟動，忽略本次姓名備援結果")
    return state


@app.post("/automation/testing")
def automation_testing(selection: ParticipantSelection) -> Dict[str, Any]:
    current = get_automation_state()
    if (current["state"] != "profile_loaded" or not current.get("participant")
            or current["participant"]["participant_id"] != selection.participant_id):
        raise HTTPException(409, "尚未完成受測者確認")
    state = _set_automation_state_if_current(
        "profile_loaded",
        current["revision"],
        "testing",
        "自動動作測試進行中",
        current["participant"],
    )
    if state is None:
        raise HTTPException(409, "自動流程已停止或重新啟動，無法開始動作測試")
    return state


@app.post("/automation/completed")
def automation_completed(selection: ParticipantSelection) -> Dict[str, Any]:
    current = get_automation_state()
    participant = current.get("participant")
    if (current["state"] != "testing" or not participant
            or participant["participant_id"] != selection.participant_id):
        raise HTTPException(409, "目前沒有此受測者正在進行的自動測試")
    state = _set_automation_state_if_current(
        "testing",
        current["revision"],
        "completed",
        "三項錄影已完成，正在分析或顯示結果",
        participant,
    )
    if state is None:
        raise HTTPException(409, "自動流程已停止或重新啟動，忽略完成訊號")
    return state


@app.post("/automation/failed")
def automation_failed(failure: AutomationFailure) -> Dict[str, Any]:
    current = get_automation_state()
    participant = current.get("participant")
    if failure.participant_id is not None:
        if not participant or participant["participant_id"] != failure.participant_id:
            raise HTTPException(409, "失敗回報不屬於目前受測者")
    state = _set_automation_state_if_current(
        failure.expected_state,
        failure.expected_revision,
        "error",
        failure.message,
        participant,
    )
    if state is None:
        raise HTTPException(409, "自動流程狀態已改變，忽略過期的失敗回報")
    return state


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康檢查：供部署時確認模型檔是否就緒。"""
    models_dir = BASE_DIR / "models"
    required = [
        "metadata.json",
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


@app.get("/exhibition/preflight")
def exhibition_preflight() -> Dict[str, Any]:
    """Return the minimum, privacy-safe checks required before an exhibit run."""
    snapshot = health()
    roster = snapshot["roster"]
    speech = snapshot["speech"]
    tts_dir = STATIC_DIR / "tts"
    missing_tts = [
        filename for filename in EXHIBITION_TTS_FILES
        if not (tts_dir / filename).is_file() or (tts_dir / filename).stat().st_size < 1_000
    ]
    required_frontend = (
        "index.html",
        "kebbi.html",
        "sit_to_stand_counter.js",
        "action_readiness.js",
    )
    missing_frontend = [
        filename for filename in required_frontend
        if not (STATIC_DIR / filename).is_file()
    ]
    checks = [
        {
            "id": "models", "label": "AI 模型", "required": True,
            "ok": not snapshot["missing_files"],
            "detail": "模型檔完整" if not snapshot["missing_files"]
            else f"缺少：{', '.join(snapshot['missing_files'])}",
        },
        {
            "id": "camera", "label": "C922 畫面", "required": True,
            "ok": bool(snapshot["camera_ready"]),
            "detail": snapshot["camera"] if snapshot["camera_ready"] else "尚未取得即時畫面",
        },
        {
            "id": "roster", "label": "受測者名冊", "required": True,
            "ok": bool(roster.get("ready")), "detail": roster.get("message", ""),
        },
        {
            "id": "frontend", "label": "時機偵測程式", "required": True,
            "ok": not missing_frontend,
            "detail": "流程與計次模組完整" if not missing_frontend
            else f"缺少：{', '.join(missing_frontend)}",
        },
        {
            "id": "tts", "label": "繁體中文預錄語音", "required": True,
            "ok": not missing_tts,
            "detail": f"{len(EXHIBITION_TTS_FILES)} 段提示音完整" if not missing_tts
            else f"缺少或無效：{', '.join(missing_tts)}",
        },
        {
            "id": "speech", "label": "姓名語音備援", "required": False,
            "ok": bool(speech.get("available")),
            "detail": speech.get("message", ""),
        },
    ]
    blockers = [check["detail"] for check in checks if check["required"] and not check["ok"]]
    warnings = [check["detail"] for check in checks if not check["required"] and not check["ok"]]
    return {
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
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
               weight_kg: float, gender: str = "",
               _authorized: None = Depends(require_clinician_token)) -> Dict[str, Any]:
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
async def save_label(
    user_id: str = Form(...),
    label: int = Form(...),
    _authorized: None = Depends(require_clinician_token),
):
    """Legacy binary label endpoint; retained only for authenticated migration."""
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

    # Formal device sessions must not reset in-memory automation state when a
    # watched file changes. Developers can still opt into reload via uvicorn.
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)

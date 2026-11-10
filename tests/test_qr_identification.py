from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import api_server
from webcam_capture import CameraHub, decode_qr_payloads


SAMPLE = [
    {
        "participant_id": "SB01",
        "name": "王小明",
        "name_aliases": "",
        "age": 75,
        "height_cm": 165.0,
        "weight_kg": 60.0,
        "gender": "male",
    }
]


@pytest.fixture(autouse=True)
def reset_automation_state():
    api_server._set_automation_state("idle", "test reset")
    yield
    api_server._set_automation_state("idle", "test reset")


def _camera_frame_with_qr(payload: str) -> bytes:
    qr = cv2.QRCodeEncoder_create().encode(payload)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (420, 420), interpolation=cv2.INTER_NEAREST)
    frame = np.full((720, 1280), 255, dtype=np.uint8)
    frame[150:570, 430:850] = qr
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def test_c922_frame_qr_decoder_returns_exact_payload():
    payload = "SARCOPENIA:SB01"
    assert decode_qr_payloads(_camera_frame_with_qr(payload)) == [payload]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("SARCOPENIA:SB01", "SB01"),
        ("sarcopenia:SB01", None),
        ("SARCOPENIA:sb01", "sb01"),
        ("SARCOPENIA:SB01|王小明", None),
        ("SB01", None),
        (" SARCOPENIA:SB01 ", None),
    ],
)
def test_qr_payload_format_is_strict_and_privacy_minimal(payload, expected):
    assert api_server.parse_qr_participant_id(payload) == expected


def test_qr_scan_exactly_matches_roster_and_loads_profile(monkeypatch):
    api_server._set_automation_state("scanning_qr", "scan")
    monkeypatch.setattr(api_server.camera, "decode_qr_codes", lambda timeout: ["SARCOPENIA:SB01"])
    monkeypatch.setattr(api_server, "read_roster_rows", lambda: SAMPLE)

    result = api_server.automation_qr_scan()

    assert result["status"] == "matched"
    assert result["participant"]["participant_id"] == "SB01"
    assert result["automation"]["state"] == "profile_loaded"


def test_qr_scan_rejects_case_mismatch_in_roster(monkeypatch):
    api_server._set_automation_state("scanning_qr", "scan")
    monkeypatch.setattr(api_server.camera, "decode_qr_codes", lambda timeout: ["SARCOPENIA:sb01"])
    monkeypatch.setattr(api_server, "read_roster_rows", lambda: SAMPLE)

    result = api_server.automation_qr_scan()

    assert result == {
        "status": "invalid",
        "message": "QR Code 編號未在名冊中找到唯一對應資料",
    }
    assert api_server.get_automation_state()["state"] == "scanning_qr"


def test_stop_during_qr_decode_cannot_resurrect_automation(monkeypatch):
    api_server._set_automation_state("scanning_qr", "scan")

    def stop_then_decode(timeout):
        api_server.stop_automation()
        return ["SARCOPENIA:SB01"]

    monkeypatch.setattr(api_server.camera, "decode_qr_codes", stop_then_decode)
    monkeypatch.setattr(api_server, "read_roster_rows", lambda: SAMPLE)

    with pytest.raises(HTTPException) as caught:
        api_server.automation_qr_scan()
    assert caught.value.status_code == 409
    assert api_server.get_automation_state()["state"] == "idle"


def test_stop_during_manual_lookup_cannot_load_profile(monkeypatch):
    api_server._set_automation_state("scanning_qr", "scan")

    def stop_then_read():
        api_server.stop_automation()
        return SAMPLE

    monkeypatch.setattr(api_server, "read_roster_rows", stop_then_read)

    with pytest.raises(HTTPException) as caught:
        api_server.automation_profile_loaded(api_server.ParticipantSelection(participant_id="SB01"))
    assert caught.value.status_code == 409
    assert api_server.get_automation_state()["state"] == "idle"


def test_camera_hub_rejects_stale_cached_frame():
    hub = object.__new__(CameraHub)
    hub.condition = threading.Condition()
    hub.latest_frame = b"old-jpeg"
    hub.latest_frame_at = time.monotonic() - 10
    hub.stop_event = threading.Event()
    hub.start = lambda: None

    assert hub.get_frame(timeout=0.01, max_age_seconds=1.0) is None

    hub.latest_frame_at = time.monotonic()
    assert hub.get_frame(timeout=0.01, max_age_seconds=1.0) == b"old-jpeg"


def test_generated_test_qr_files_contain_only_expected_identifiers():
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "automation-qr"
    detector = cv2.QRCodeDetector()
    decoded = {}
    for path in sorted(output_dir.glob("SB*.png")):
        value, _, _ = detector.detectAndDecode(cv2.imread(str(path)))
        decoded[path.name] = value
    assert decoded == {
        "SB01.png": "SARCOPENIA:SB01",
        "SB02.png": "SARCOPENIA:SB02",
        "SB03.png": "SARCOPENIA:SB03",
    }

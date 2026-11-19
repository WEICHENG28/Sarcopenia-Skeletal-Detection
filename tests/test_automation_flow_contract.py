from pathlib import Path

import pytest
from fastapi import HTTPException

import api_server


INDEX_PATH = Path(api_server.__file__).resolve().parent / "static" / "index.html"
SAMPLE = {
    "participant_id": "SB01",
    "name": "王小明",
    "name_aliases": "",
    "age": 75,
    "height_cm": 165.0,
    "weight_kg": 60.0,
    "gender": "male",
}


@pytest.fixture(autouse=True)
def reset_automation_state():
    api_server._set_automation_state("idle", "test reset")
    yield
    api_server._set_automation_state("idle", "test reset")


def test_start_opens_qr_scan_immediately(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "participant_roster_status",
        lambda: {"ready": True, "message": "ready"},
    )
    monkeypatch.setattr(api_server.camera, "get_frame", lambda timeout: b"jpeg")

    state = api_server.start_automation()

    assert state["state"] == "scanning_qr"
    assert "QR Code 掃描" in state["message"]


def test_qr_identity_state_machine_reaches_testing_and_completed(monkeypatch):
    api_server._set_automation_state("scanning_qr", "scan identity first")
    monkeypatch.setattr(
        api_server.camera,
        "decode_qr_codes",
        lambda timeout: ["SARCOPENIA:SB01"],
    )
    monkeypatch.setattr(api_server, "read_roster_rows", lambda: [SAMPLE])

    scanned = api_server.automation_qr_scan()
    assert scanned["status"] == "matched"
    assert scanned["automation"]["state"] == "profile_loaded"
    testing = api_server.automation_testing(
        api_server.ParticipantSelection(participant_id="SB01")
    )
    assert testing["state"] == "testing"
    completed = api_server.automation_completed(
        api_server.ParticipantSelection(participant_id="SB01")
    )
    assert completed["state"] == "completed"


def test_completion_rejects_wrong_participant():
    api_server._set_automation_state("testing", "running", SAMPLE)
    with pytest.raises(HTTPException) as caught:
        api_server.automation_completed(
            api_server.ParticipantSelection(participant_id="SB99")
        )
    assert caught.value.status_code == 409
    assert api_server.get_automation_state()["state"] == "testing"


def test_stale_failure_cannot_overwrite_a_stopped_or_restarted_flow():
    running = api_server._set_automation_state("testing", "running", SAMPLE)
    failure = api_server.AutomationFailure(
        message="late upload failure",
        expected_state="testing",
        expected_revision=running["revision"],
        participant_id="SB01",
    )
    api_server.stop_automation()

    with pytest.raises(HTTPException) as caught:
        api_server.automation_failed(failure)
    assert caught.value.status_code == 409
    assert api_server.get_automation_state()["state"] == "idle"


def test_frontend_contract_covers_full_body_actions_and_failure_fallback():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "if (nowSec - automationPresenceSince < 2.5) return;" in source
    assert "handleAutomationPresence(presenceFramingOk, nowSec)" in source
    assert "PERSON_ENTRY_STABLE_SEC" not in source
    assert "function checkPersonPresent(landmarks)" not in source
    assert "automationServerState.state !== 'profile_loaded'" in source
    assert "void beginAutomatedTests(automationServerState.participant);" in source
    assert "const REP_TARGET = { sit: 5 };" in source
    assert "autoStopCurrentRecording('已到達終點定位')" in source
    assert "const SQUAT_RECORD_MS = 4000;" in source
    assert "autoStopCurrentRecording('半蹲錄影 4 秒完成')" in source
    assert "if (automated) void reportAutomationFailure(failureMessage, false);" in source
    assert "lastResult?.inference_result?.status !== 'success'" in source
    assert "const WALK_MAX_RECORD_MS = 30000;" in source
    assert "actionId === 'walk' && !stoppedActionCompleted" in source
    assert "本次沒有有效的 AI 風險結果" in source
    assert "不是依本次檢測產生的建議" in source

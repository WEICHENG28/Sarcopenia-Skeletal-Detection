"""自動偵測模式：有人走近就開始，誤入時可中止或自動回到待命。"""

from pathlib import Path

import pytest
from fastapi import HTTPException

import api_server


INDEX_PATH = Path(api_server.__file__).resolve().parent / "static" / "index.html"
CONTROL_PATH = INDEX_PATH.with_name("control.html")


@pytest.fixture(autouse=True)
def reset_automation_state(monkeypatch):
    monkeypatch.setattr(
        api_server, "participant_roster_status", lambda: {"ready": True, "message": "ready"}
    )
    monkeypatch.setattr(api_server.camera, "get_frame", lambda timeout: b"jpeg")
    api_server.disarm_automation()
    api_server._set_automation_state("idle", "test reset")
    yield
    api_server.disarm_automation()
    api_server._set_automation_state("idle", "test reset")


def test_arm_enters_standby_and_person_detection_starts_a_round():
    armed = api_server.arm_automation()
    assert armed["state"] == "armed"
    assert armed["auto_mode"] is True

    started = api_server.automation_person_detected()
    assert started["state"] == "scanning_qr"


def test_stop_returns_to_standby_in_auto_mode_and_to_idle_otherwise():
    api_server.arm_automation()
    api_server.automation_person_detected()
    assert api_server.stop_automation()["state"] == "armed"

    api_server.disarm_automation()
    api_server.start_automation()
    assert api_server.stop_automation()["state"] == "idle"


def test_standby_requires_auto_mode_and_the_current_revision():
    api_server.start_automation()
    current = api_server.get_automation_state()
    request = api_server.AutomationStandby(
        message="受測者已離開，回到待命",
        expected_state="scanning_qr",
        expected_revision=current["revision"],
    )
    with pytest.raises(HTTPException):
        api_server.automation_standby(request)

    api_server.arm_automation()
    api_server.automation_person_detected()
    current = api_server.get_automation_state()
    stale = request.model_copy(update={"expected_revision": current["revision"] - 1})
    with pytest.raises(HTTPException):
        api_server.automation_standby(stale)

    fresh = request.model_copy(update={"expected_revision": current["revision"]})
    assert api_server.automation_standby(fresh)["state"] == "armed"


def test_disarm_does_not_interrupt_a_running_round():
    api_server.arm_automation()
    api_server.automation_person_detected()
    state = api_server.disarm_automation()
    assert state["state"] == "scanning_qr"
    assert state["auto_mode"] is False


def test_kebbi_page_auto_starts_only_for_a_near_still_person():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "function handleAutoStart(landmarks, nowSec)" in source
    assert "const near = personInView(landmarks, AUTO_START_MIN_SHOULDER_SPAN);" in source
    # 空房間的誤判小骨架不算有人，否則無法淨空、也無法因人離開而回到待命
    assert "if (personInView(landmarks)) lastPersonSeenSec = nowSec;" in source
    assert "if (!personInView(landmarks)) {" in source
    assert "requestJson('/automation/person-detected', { method: 'POST' })" in source
    # 回到待命後畫面要先淨空，剛被中止的人不會立刻再觸發
    assert "if (nowSec - lastPersonSeenSec >= AUTO_CLEAR_SEC) armedNeedsClear = false;" in source


def test_kebbi_page_returns_to_standby_after_no_qr_or_absence():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const AUTO_CANCEL_NO_QR_MS = 30000;" in source
    assert "const AUTO_CANCEL_ABSENT_SEC = 10;" in source
    assert "void returnToAutoStandby('30 秒內沒有掃到 QR Code，回到待命');" in source
    assert "void returnToAutoStandby('受測者已離開，回到待命');" in source
    # 結果顯示後才回到待命，避免分析途中清掉畫面
    assert "state.state === 'completed' && automationResultShownAt !== null" in source


def test_each_round_starts_with_the_greeting():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "void queueSpeech(KEBBI_GREETING, { minimumMs: 300 });" in source
    assert source.index("void queueSpeech(KEBBI_GREETING") < source.index("void queueSpeech(QR_PROMPT")
    assert "new Audio('/static/tts/chime.wav')" in source


def test_control_page_has_auto_detection_switch():
    control = CONTROL_PATH.read_text(encoding="utf-8")
    assert "onclick=\"toggleAutoMode()\"" in control
    assert "'/automation/disarm' : '/automation/arm'" in control

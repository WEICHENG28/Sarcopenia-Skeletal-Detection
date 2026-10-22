from pathlib import Path


INDEX_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"
TTS_PATH = INDEX_PATH.parent / "tts"


def test_tts_has_timeout_and_minimum_wait_fallbacks():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "TTS_TIMEOUT_MIN_MS" in source
    assert "TTS_TIMEOUT_MAX_MS" in source
    assert "語音播放逾時，流程自動繼續" in source
    assert "minimumMs: 1000, timeoutMs: 2200" in source


def test_profile_loaded_state_can_resume_after_page_reload():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "state.state === 'profile_loaded' && state.participant" in source
    assert "loadParticipantIntoForm(state.participant)" in source
    assert "beginAutomatedTests(state.participant)" in source


def test_kebbi_page_requests_screen_wake_lock_and_polls_on_resume():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "navigator.wakeLock.request('screen')" in source
    assert "document.addEventListener('visibilitychange'" in source
    assert "ensureScreenWakeLock();" in source
    assert "pollAutomationState();" in source


def test_recorded_traditional_chinese_prompts_are_packaged():
    source = INDEX_PATH.read_text(encoding="utf-8")
    expected = {
        "manual_name.mp3",
        "profile_loaded.mp3",
        "sit_prompt.mp3",
        "walk_prompt.mp3",
        "squat_prompt.mp3",
        "count_1.mp3",
        "count_2.mp3",
        "count_3.mp3",
        "count_4.mp3",
        "count_5.mp3",
        "start.mp3",
        "enter_frame.mp3",
        "hold.mp3",
        "good.mp3",
        "complete_5.mp3",
        "movement_stopped.mp3",
        "analysis.mp3",
    }
    assert {path.name for path in TTS_PATH.glob("*.mp3")} == expected
    assert all((TTS_PATH / filename).stat().st_size > 1_000 for filename in expected)
    assert "RECORDED_PROMPTS" in source
    assert "提示音檔播放失敗，改用瀏覽器 TTS" in source

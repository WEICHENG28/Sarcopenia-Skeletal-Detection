from pathlib import Path


INDEX_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"
TTS_PATH = INDEX_PATH.parent / "tts"


def test_tts_has_timeout_and_minimum_wait_fallbacks():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "TTS_TIMEOUT_MIN_MS" in source
    assert "TTS_TIMEOUT_MAX_MS" in source
    assert "語音播放逾時，流程自動繼續" in source
    assert "minimumMs: 1000, timeoutMs: 2200" in source
    assert "RECORDED_AUDIO_TIMEOUT_BUFFER_MS" in source
    assert "promptAudio.onloadedmetadata" in source
    assert "Math.ceil(promptAudio.duration * 1000)" in source


def test_simulation_does_not_force_short_timeout_on_long_action_prompts():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "await speakAsync(action.prompt);" in source
    assert "await speakAsync(action.prompt, { timeoutMs: 5000 });" not in source


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


def test_only_visible_kebbi_tab_controls_automation():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "AUTOMATION_TAB_STORAGE_KEY" in source
    assert "AUTOMATION_PAGE_INSTANCE_ID" in source
    assert "claimAutomationTab();" in source
    assert "window.addEventListener('storage'" in source
    assert "if (!isActiveAutomationTab) return;" in source
    assert "此分頁已停止控制" in source
    assert "if (document.visibilityState === 'visible') claimAutomationTab();" in source
    assert "if (!isActiveAutomationTab) return;" in source


def test_recorded_traditional_chinese_prompts_are_packaged():
    source = INDEX_PATH.read_text(encoding="utf-8")
    expected = {
        "manual_name.mp3",
        "ask_name.mp3",
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
        "end.mp3",
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


def test_gecko_falls_back_to_pc_side_offline_speech_recognition():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "recognizeWithServer" in source
    assert "'/speech/listen'" in source
    assert "purpose === 'confirmation' ? 3.5 : 4.5" in source
    assert "已自動改用電腦端 C922 麥克風與離線模型" in source


def test_robot_speech_is_suppressed_while_c922_is_listening():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "POST_TTS_LISTEN_DELAY_MS" in source
    assert "speechCaptureActive = true" in source
    assert "if (speechCaptureActive || !speechEnabled" in source
    assert "currentRecordingActionId || automationServerState.state === 'armed'" in source


def test_kebbi_listening_starts_promptly_and_beeps_after_recorder_start():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const POST_TTS_LISTEN_DELAY_MS = 800;" in source
    assert "function playListeningBeep()" in source
    assert "recorder.start(250);" in source
    assert "await playListeningBeep();" in source
    assert source.index("recorder.start(250);") < source.index("await playListeningBeep();")
    assert "聽到嗶聲後，請清楚說話" in source


def test_name_question_uses_the_matching_prerecorded_prompt_before_listening():
    source = INDEX_PATH.read_text(encoding="utf-8")
    prompt = "您好，請在嗶聲後，只說您的姓名，例如王小明。"
    assert f"'{prompt}': '/static/tts/ask_name.mp3?v=2'" in source
    assert f"'{prompt}',\n                        'name'" in source
    assert source.index("await speakAsync(promptText);") < source.index(
        "await new Promise(resolve => setTimeout(resolve, POST_TTS_LISTEN_DELAY_MS));"
    )
    assert source.index("await speakAsync(promptText);") < source.index(
        "return await recognizeWithKebbiMicrophone(purpose)"
    )


def test_kebbi_microphone_is_preferred_with_c922_fallback():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "navigator.mediaDevices.getUserMedia" in source
    assert "new MediaRecorder" in source
    assert "echoCancellation: true" in source
    assert "noiseSuppression: true" in source
    assert "/speech/transcribe-upload?purpose=${encodeURIComponent(purpose)}" in source
    assert "return await recognizeWithKebbiMicrophone(purpose)" in source
    assert "已改用 C922 麥克風" in source
    assert "stream.getTracks().forEach(track => track.stop())" in source
    assert "if (isSpeechBusyError(error)) throw error;" in source


def test_formal_name_flow_asks_for_direct_name_and_stops_without_manual_input():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "只說您的姓名，例如王小明" in source
    assert "ALLOW_MANUAL_NAME_FALLBACK" in source
    assert "姓名辨識失敗，本次檢測已停止" in source
    assert "語音服務仍在處理上一段錄音" in source


def test_pc_rehearsal_mode_is_clearly_labeled_and_forces_c922_speech():
    source = INDEX_PATH.read_text(encoding="utf-8")
    control = (INDEX_PATH.parent / "control.html").read_text(encoding="utf-8")
    assert "PC_REHEARSAL_MODE" in source
    assert "電腦彩排模式" in source
    assert "不代表凱比實機測試完成" in source
    assert "if (PC_REHEARSAL_MODE) return recognizeWithServer(purpose);" in source
    assert 'href="/?rehearsal=1"' in control
    assert "沒有凱比：開啟電腦彩排畫面" in control


def test_no_hardware_simulation_is_explicit_and_skips_camera_startup():
    source = INDEX_PATH.read_text(encoding="utf-8")
    control = (INDEX_PATH.parent / "control.html").read_text(encoding="utf-8")
    assert "NO_HARDWARE_SIMULATION_MODE" in source
    assert "runNoHardwareSimulation" in source
    assert "不得作為實機或模型成果" in source
    assert "未執行錄影、計次或模型推論" in source
    assert "if (NO_HARDWARE_SIMULATION_MODE)" in source
    assert 'href="/?rehearsal=1&amp;mock=1"' in control
    assert "沒有凱比也沒有Webcam：開啟流程模擬" in control


def test_each_action_announces_end_before_advancing():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "'結束': '/static/tts/end.mp3'" in source
    assert "setBanner(`${ACTION_LABEL[finishedActionId]}：結束`, 'ok')" in source
    assert "await speakAsync('結束', { minimumMs: 800 });" in source

from pathlib import Path


INDEX_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"
COUNTER_PATH = INDEX_PATH.parent / "sit_to_stand_counter.js"
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


def test_profile_loaded_state_waits_for_full_body_after_page_reload():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "state.state === 'profile_loaded' && state.participant" in source
    assert "loadParticipantIntoForm(state.participant)" in source
    assert "不可直接跳進動作測試" in source
    assert "void beginAutomatedTests(automationServerState.participant);" in source


def test_kebbi_page_requests_screen_wake_lock_and_polls_on_resume():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "navigator.wakeLock.request('screen')" in source
    assert "document.addEventListener('visibilitychange'" in source
    assert "ensureScreenWakeLock();" in source
    assert "pollAutomationState();" in source


def test_kebbi_mode_attempts_audio_automatically_with_tap_fallback():
    source = INDEX_PATH.read_text(encoding="utf-8")
    launcher = (INDEX_PATH.parents[1] / "scripts" / "start_kebbi_session.ps1").read_text(
        encoding="utf-8"
    )
    assert "const KEBBI_MODE = PAGE_PARAMS.get('kebbi') === '1';" in source
    assert 'id="audioUnlockOverlay"' in source
    assert "if (KEBBI_MODE) void attemptKebbiAudioAutostart();" in source
    assert "async function attemptKebbiAudioAutostart()" in source
    assert "async function enableKebbiAudio(button, automatic = false)" in source
    assert "await confirmation.play();" in source
    assert "if (!kebbiAudioReady)" in source
    assert "http://127.0.0.1:8000/?kebbi=1" in launcher


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
        "sit_ready.mp3",
        "walk_ready.mp3",
        "squat_ready.mp3",
        "pose_ready.mp3",
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
        "greeting.mp3",
        "qr_prompt.mp3",
        "complete_5.mp3",
        "movement_stopped.mp3",
        "squat_recorded.mp3",
        "walk_arrived.mp3",
        "retry.mp3",
        "analysis.mp3",
    }
    assert {path.name for path in TTS_PATH.glob("*.mp3")} == expected
    assert all((TTS_PATH / filename).stat().st_size > 1_000 for filename in expected)
    assert "RECORDED_PROMPTS" in source
    assert "提示音檔播放失敗，改用瀏覽器 TTS" in source


def test_optional_speech_helper_still_supports_pc_side_offline_recognition():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "recognizeWithServer" in source
    assert "'/speech/listen'" in source
    assert "purpose === 'confirmation' ? 3.5 : 4.5" in source


def test_robot_speech_is_suppressed_while_c922_is_listening():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "POST_TTS_LISTEN_DELAY_MS" in source
    assert "speechCaptureActive = true" in source
    assert "if (speechCaptureActive || !speechEnabled" in source
    assert "currentRecordingActionId || automationServerState.state === 'profile_loaded'" in source


def test_kebbi_listening_starts_promptly_and_beeps_after_recorder_start():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const POST_TTS_LISTEN_DELAY_MS = 800;" in source
    assert "function playListeningBeep()" in source
    assert "recorder.start(250);" in source
    assert "await playListeningBeep();" in source
    assert source.index("recorder.start(250);") < source.index("await playListeningBeep();")
    assert "聽到嗶聲後，請清楚說話" in source


def test_name_speech_prompt_is_not_invoked_by_formal_identity_flow():
    source = INDEX_PATH.read_text(encoding="utf-8")
    prompt = "您好，請在嗶聲後，只說您的姓名，例如王小明。"
    assert f"'{prompt}': '/static/tts/ask_name.mp3?v=2'" in source
    assert source.count(prompt) == 1
    assert "await speakAsync('請將受測者 QR Code 對準攝影機。');" not in source


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


def test_formal_identity_flow_uses_server_side_qr_with_manual_name_fallback():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "startQrIdentityFlow" in source
    assert "'/automation/qr-scan'" in source
    assert "MANUAL_NAME_FALLBACK_DELAY_MS = 12000" in source
    assert "QR 僅含受測者編號" in source
    assert "以姓名備援確認" in source
    assert "startIdentityConversation" not in source
    assert "const fallbackTimer = setTimeout" in source
    assert "'/automation/qr-scan', { method: 'POST' }, 2500" in source
    assert "controller.abort()" in source


def test_new_qr_cycle_clears_previous_participant_before_scanning():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "state.state === 'scanning_qr'" in source
    assert "if (previousState !== 'scanning_qr') prepareAutomationCycle();" in source
    assert "避免展場畫面" in source


def test_pc_rehearsal_mode_is_clearly_labeled_and_keeps_c922_qr_flow():
    source = INDEX_PATH.read_text(encoding="utf-8")
    control = (INDEX_PATH.parent / "control.html").read_text(encoding="utf-8")
    assert "PC_REHEARSAL_MODE" in source
    assert "電腦彩排模式" in source
    assert "不代表凱比實機測試完成" in source
    assert "身分仍由電腦端 C922 掃描 QR Code" in source
    assert "'/automation/qr-scan'" in source
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


def test_each_action_ends_with_a_single_completion_announcement():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "setBanner(`${ACTION_LABEL[finishedActionId]}：完成`, 'ok')" in source
    # 完成語句由各動作停止時播報，進入下一項前只等待播完，不再重複說「結束」
    advance = source[source.index("async function advanceAutomatedTest("):]
    advance = advance[:advance.index("automationActionIndex++;")]
    assert "await liveSpeechChain;" in advance
    assert "speakAsync('結束'" not in advance


def test_result_page_includes_safe_rehabilitation_education():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "檢測後一般活動與防跌衛教" in source
    assert "不是個別運動處方" in source
    assert "穩固、無輪子的椅子" in source
    assert "每週至少 2 天" in source
    assert "每週 3 天以上為一般目標" in source
    assert "胸痛／胸悶" in source
    assert "緊急時撥 119" in source
    assert "同一肌群兩次間至少休息 1 天" in source
    assert "無法達到一般建議量時" in source
    assert "rehabNextStep" in source
    assert "renderRehabEducation(serverLevel || level, inferenceAvailable)" in source
    assert "每公斤體重約 1.2g/天" not in source
    assert "補充維生素 D" not in source


def test_live_feedback_is_queued_instead_of_cutting_off_previous_speech():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "let liveSpeechChain = Promise.resolve();" in source
    assert "function queueSpeech(text" in source
    assert "liveSpeechChain = liveSpeechChain.then(() => speakAsync(text, options));" in source
    assert "void queueSpeech(text);" in source
    assert ".then(() => autoStopCurrentRecording('已完成目標次數'))" in source


def test_sit_to_stand_counts_stable_seated_to_standing_transitions():
    source = INDEX_PATH.read_text(encoding="utf-8")
    counter = COUNTER_PATH.read_text(encoding="utf-8")
    assert "const REP_TARGET = { sit: 5 };" in source
    assert '<script src="/static/sit_to_stand_counter.js"></script>' in source
    assert "SitToStandCounter.computeMetrics(landmarks)" in source
    assert "onRepCounted('sit', result.repCount)" in source
    assert "stableFrames: 2" in counter
    assert "standingKneeMin: 150" in counter
    assert "standingRiseTorso: 0.22" in counter
    assert "calibrationSeatedThighMin: 0.18" in counter
    assert "state.smoothedThighHorizontalTorso" in counter
    assert "candidate === 'up' && state.phase === 'down'" in counter
    assert "state.baselineKneeAngle" in counter
    assert "REP_CALIB_WINDOW_SEC" not in source


def test_sit_to_stand_allows_one_visible_ankle_after_full_body_entry_check():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "handleAutomationPresence(presenceFramingOk, nowSec)" in source
    assert "checkSitToStandFraming(landmarks)" in source
    assert ".some(i => landmarkIsInFrame(landmarks[i]))" in source


def test_walk_stops_at_a_stable_endpoint_displacement():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "走向地面標示的終點" in source
    assert "WALK_TARGET_DISPLACEMENT = 0.16" in source
    assert "WALK_TARGET_STABLE_FRAMES = 2" in source
    assert "live.walkTargetFrames < WALK_TARGET_STABLE_FRAMES" in source
    assert "autoStopCurrentRecording('已到達終點定位')" in source


def test_squat_records_one_repetition_for_four_seconds():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "聽到開始後做一次半蹲" in source
    assert "SQUAT_RECORD_MS = 4000" in source
    assert "setTimeout(() =>" in source
    assert "autoStopCurrentRecording('半蹲錄影 4 秒完成')" in source
    assert "squat: 5" not in source
    assert "[AUTOMATED_ACTIONS[2].prompt]: '/static/tts/squat_prompt.mp3'" in source


def test_each_action_waits_for_pose_readiness_before_countdown():
    source = INDEX_PATH.read_text(encoding="utf-8")
    readiness = (INDEX_PATH.parent / "action_readiness.js").read_text(encoding="utf-8")
    assert '<script src="/static/action_readiness.js"></script>' in source
    assert "await speakAsync(step.readyPrompt" in source
    assert "const ready = await waitForActionReadiness(step, token);" in source
    assert source.index("const ready = await waitForActionReadiness(step, token);") < source.index(
        "for (let count = 3; count >= 1; count--)"
    )
    assert "updatePendingActionReadiness(landmarks);" in source
    assert "READINESS_OVERRIDE_DELAY_MS = 12000" in source
    assert 'onclick="confirmActionReadiness()"' in source
    assert "stableFrames: 5" in readiness
    assert "seatedKneeMax: 155" in readiness
    assert "standingKneeMin: 150" in readiness


def test_flow_timeline_covers_identity_motion_and_result_stages():
    source = INDEX_PATH.read_text(encoding="utf-8")
    for label in [">QR<", ">入鏡<", ">坐站<", ">行走<", ">半蹲<", ">結果<"]:
        assert label in source
    assert "function updateFlowTimeline" in source
    assert "aria-current" in source


def test_control_console_blocks_start_until_exhibition_preflight_passes():
    control = (INDEX_PATH.parent / "control.html").read_text(encoding="utf-8")
    assert "展場預檢" in control
    assert "api('/exhibition/preflight')" in control
    assert "!preflight.ready || busy" in control
    assert "if (!preflightReady)" in control


def test_upload_failure_reports_server_detail_instead_of_guessing_server_is_off():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "detail = (await res.json()).detail || detail" in source
    assert "上傳或分析失敗" in source
    assert "本頁仍提供一般性安全衛教" in source
    assert "showResult(sid" in source
    assert "請確認後端伺服器是否已啟動" not in source


def test_failed_or_skipped_inference_is_not_presented_as_low_risk():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "infer.status === 'success'" in source
    assert "Number.isFinite(rawProbability)" in source
    assert "inferenceAvailable ? `${pct}%` : '--'" in source
    assert "renderRehabEducation(serverLevel || level, inferenceAvailable)" in source
    assert "box.style.display = 'block'" in source
    assert "本次沒有有效的 AI 風險結果" in source
    assert "不是依本次檢測產生的建議" in source


def test_statement_level_calls_only_use_defined_functions():
    # 合併衝突曾留下呼叫已刪除函式的程式碼，導致每次開始錄影都丟出 ReferenceError。
    import re

    source = INDEX_PATH.read_text(encoding="utf-8")
    called = set(re.findall(
        r"^\s+(?:await\s+|void\s+|return\s+)?([a-z]\w*)\(", source, re.M
    ))
    defined = set(re.findall(r"function\s+(\w+)", source))
    defined |= set(re.findall(r"(?:const|let|var)\s+(\w+)\s*=", source))
    browser_globals = {
        "alert", "cancelAnimationFrame", "clearInterval", "clearTimeout",
        "reject", "resolve", "setInterval", "setTimeout",
    }
    assert called - defined - browser_globals == set()


def test_webcam_preview_restarts_when_page_becomes_visible_again():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "function loadNextWebcamFrame()" in source
    assert "if (claimAutomationTab()) loadNextWebcamFrame();" in source


def test_failed_action_is_retried_within_the_same_automated_round():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const MAX_AUTOMATED_ACTION_RETRIES = 2;" in source
    assert source.count("if (retryCurrentAutomatedAction(actionId)) return;") == 2
    assert "automationRetryCounts = {};" in source
    assert "await advanceAutomatedTest(null, token, { retry: true });" in source
    assert "[ACTION_RETRY_PROMPT]: '/static/tts/retry.mp3'" in source


def test_result_page_hides_score_when_uncertain_or_below_awgs_age():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "if (infer.age_not_applicable) {" in source
    assert "不適用（未滿 60 歲）" in source
    assert "probEl.style.display = infer.uncertain || infer.age_not_applicable ? 'none' : '';" in source


def test_walk_requires_standing_still_and_fails_when_leaving_frame():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const WALK_STILL_MOTION = 0.012;" in source
    assert "&& motion <= WALK_STILL_MOTION" in source
    assert "const WALK_OUT_OF_FRAME_FAIL_SEC = 2.0;" in source
    assert "void autoStopCurrentRecording('走出畫面', 'warn');" in source
    # 到達後立即停止錄影，不等語音播完
    assert ".then(() => autoStopCurrentRecording('已到達終點定位'))" not in source


def test_sit_posture_is_calibrated_before_the_start_cue():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "let preparedSitCounter = null;" in source
    assert "preparedSitCounter.update(window.SitToStandCounter.computeMetrics(landmarks));" in source
    assert "const prepared = takePreparedSitCounter();" in source
    assert "baselineHipY !== null && phase === 'down' && repCount === 0" in source


def test_kebbi_greeting_is_recorded_and_not_the_old_confirmation():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "[KEBBI_GREETING]: '/static/tts/greeting.mp3'" in source
    assert "new Audio('/static/tts/good.mp3')" not in source


def test_qr_stage_asks_for_the_qr_code_without_blocking_the_scan():
    source = INDEX_PATH.read_text(encoding="utf-8")
    assert "const QR_PROMPT = '請給我看您的 QR Code。';" in source
    assert "[QR_PROMPT]: '/static/tts/qr_prompt.mp3'" in source
    assert "void queueSpeech(QR_PROMPT, { minimumMs: 300 });" in source

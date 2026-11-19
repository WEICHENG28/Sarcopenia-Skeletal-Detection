import api_server


def test_legacy_kebbi_launcher_redirects_to_formal_flow():
    source = (api_server.STATIC_DIR / "kebbi.html").read_text(encoding="utf-8")

    assert "window.location.replace('/?kebbi=1')" in source
    assert "url=/?kebbi=1" in source


def test_exhibition_preflight_requires_camera_models_roster_frontend_and_tts(monkeypatch):
    monkeypatch.setattr(api_server.camera, "get_frame", lambda timeout=0.1: b"jpeg")
    monkeypatch.setattr(
        api_server,
        "participant_roster_status",
        lambda: {"ready": True, "count": 3, "message": "名冊可使用"},
    )
    monkeypatch.setattr(
        api_server.speech_recognizer,
        "status",
        lambda: {"available": False, "message": "姓名語音備援未載入"},
    )

    result = api_server.exhibition_preflight()

    assert result["ready"] is True
    assert result["blockers"] == []
    assert result["warnings"] == ["姓名語音備援未載入"]
    assert {item["id"] for item in result["checks"]} == {
        "models", "camera", "roster", "frontend", "tts", "speech"
    }


def test_exhibition_preflight_blocks_when_a_required_prompt_is_missing(monkeypatch):
    monkeypatch.setattr(api_server.camera, "get_frame", lambda timeout=0.1: b"jpeg")
    monkeypatch.setattr(
        api_server,
        "participant_roster_status",
        lambda: {"ready": True, "count": 3, "message": "名冊可使用"},
    )
    monkeypatch.setattr(
        api_server.speech_recognizer,
        "status",
        lambda: {"available": True, "message": "離線語音辨識可使用"},
    )
    monkeypatch.setattr(api_server, "EXHIBITION_TTS_FILES", ("does-not-exist.mp3",))

    result = api_server.exhibition_preflight()

    assert result["ready"] is False
    assert any("does-not-exist.mp3" in blocker for blocker in result["blockers"])
    tts_check = next(item for item in result["checks"] if item["id"] == "tts")
    assert tts_check["required"] is True
    assert tts_check["ok"] is False

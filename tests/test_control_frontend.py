from pathlib import Path

from api_server import serve_control, serve_index
import api_server


CONTROL_PATH = Path(__file__).resolve().parents[1] / "static" / "control.html"


def test_control_page_uses_qr_as_primary_identity_and_keeps_speech_fallback():
    source = CONTROL_PATH.read_text(encoding="utf-8")
    assert "掃描受測者編號 QR Code" in source
    assert "正式身分確認" in source
    assert "測試姓名備援" in source
    assert "scanning_qr: '掃描 QR Code'" in source
    # 自動偵測待命（armed）時仍可手動開始，所以不列入忙碌狀態
    assert "['scanning_qr','identifying','profile_loaded','testing']" in source
    assert "api('/speech/status')" in source
    assert "api('/speech/listen'" in source
    assert "purpose: 'name'" in source
    assert "api('/participants/match'" in source
    assert "failureMessage" in source
    assert source.count("document.getElementById('errorText').textContent = '';") == 1


def test_kebbi_and_control_pages_disable_browser_cache():
    expected = "no-store, no-cache, must-revalidate"
    assert serve_index().headers["cache-control"] == expected
    assert serve_control().headers["cache-control"] == expected


def test_formal_server_entrypoint_does_not_enable_hot_reload():
    source = Path(api_server.__file__).read_text(encoding="utf-8")
    assert 'uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)' in source

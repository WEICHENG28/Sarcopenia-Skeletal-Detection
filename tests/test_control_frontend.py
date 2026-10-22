from pathlib import Path


CONTROL_PATH = Path(__file__).resolve().parents[1] / "static" / "control.html"


def test_control_page_reports_and_tests_offline_speech():
    source = CONTROL_PATH.read_text(encoding="utf-8")
    assert "測試姓名語音" in source
    assert "api('/speech/status')" in source
    assert "api('/speech/listen'" in source
    assert "purpose: 'name'" in source
    assert "api('/participants/match'" in source
    assert "failureMessage" in source

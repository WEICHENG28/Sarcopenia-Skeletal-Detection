from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_one_click_launcher_starts_hidden_server_and_opens_kebbi_flow():
    batch = (ROOT / "start_kebbi.bat").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "start_kebbi_session.ps1").read_text(encoding="utf-8")

    assert "start_kebbi_session.ps1" in batch
    assert "-WindowStyle Hidden" in script
    assert '"http://127.0.0.1:8000/health"' in script
    assert "reverse tcp:8000 tcp:8000" in script
    assert '"http://127.0.0.1:8000/?kebbi=1"' in script
    assert 'Start-Process "http://localhost:8000/control"' in script


def test_launcher_prefers_kebbi_app_over_firefox():
    script = (ROOT / "scripts" / "start_kebbi_session.ps1").read_text(encoding="utf-8")
    assert '$kebbiApp = "tw.sarcopenia.kebbi"' in script
    assert "am force-stop org.mozilla.firefox" in script
    assert 'am start -n "$kebbiApp/.MainActivity"' in script
    assert "shell monkey" not in script


def test_launcher_does_not_add_breeze_dependency():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").casefold()
    launcher = (ROOT / "scripts" / "start_kebbi_session.ps1").read_text(
        encoding="utf-8"
    ).casefold()

    assert "breeze" not in requirements
    assert "breeze" not in launcher

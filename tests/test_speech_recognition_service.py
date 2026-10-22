from pathlib import Path
from types import SimpleNamespace

from speech_recognition_service import OfflineSpeechRecognizer


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([FakeSegment(" Adam Wang ")]), SimpleNamespace(
            language="en", language_probability=0.99
        )


def test_capture_command_uses_c922_mic_and_private_pcm_wav(tmp_path):
    destination = tmp_path / "speech.wav"
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", microphone_name="C922 test microphone"
    )
    command = recognizer.capture_command(destination, 4.5)
    assert "audio=C922 test microphone" in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[-1] == str(destination)


def test_listen_transcribes_locally_and_deletes_temporary_audio(monkeypatch):
    model = FakeModel()
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", model_factory=lambda *args, **kwargs: model
    )
    captured_paths: list[Path] = []

    def fake_capture(destination: Path, duration_seconds: float) -> None:
        captured_paths.append(destination)
        destination.write_bytes(b"RIFF" + b"0" * 2_000)

    monkeypatch.setattr(recognizer, "_capture", fake_capture)
    result = recognizer.listen(
        purpose="name", roster_hints=["Adam Wang", "Adam"], duration_seconds=4.5
    )

    assert result["transcripts"] == ["Adam Wang"]
    assert result["language"] == "en"
    assert model.calls[0][1]["hotwords"] == "Adam Wang，Adam"
    assert captured_paths and not captured_paths[0].exists()


def test_confirmation_prompt_constrains_yes_and_no(monkeypatch):
    model = FakeModel()
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", model_factory=lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        recognizer,
        "_capture",
        lambda destination, duration: destination.write_bytes(b"RIFF" + b"0" * 2_000),
    )

    recognizer.listen(purpose="confirmation", roster_hints=[], duration_seconds=3.5)
    options = model.calls[0][1]
    assert "不是" in options["initial_prompt"]
    assert "yes no" in options["hotwords"]


def test_model_uses_cache_first_then_allows_one_time_download():
    calls: list[bool] = []
    model = FakeModel()

    def factory(*args, **kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise RuntimeError("cache miss")
        return model

    recognizer = OfflineSpeechRecognizer("ffmpeg.exe", model_factory=factory)
    assert recognizer._ensure_model() is model
    assert calls == [True, False]

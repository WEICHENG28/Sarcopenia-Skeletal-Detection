import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def test_kebbi_prompt_echo_is_not_treated_as_a_name():
    assert OfflineSpeechRecognizer._is_robot_name_prompt("請完整﹑") is True
    assert OfflineSpeechRecognizer._is_robot_name_prompt("請完整入鏡") is True
    assert OfflineSpeechRecognizer._is_robot_name_prompt("只說您的姓名") is True
    assert OfflineSpeechRecognizer._is_robot_name_prompt("例如王小明") is True
    assert OfflineSpeechRecognizer._is_robot_name_prompt("王小明") is False
    assert OfflineSpeechRecognizer._is_robot_name_prompt("我的名字是王小明") is False


def test_listen_transcribes_locally_and_deletes_temporary_audio(monkeypatch):
    model = FakeModel()
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", model_factory=lambda *args, **kwargs: model
    )
    captured_paths: list[Path] = []

    def fake_capture(destination: Path, duration_seconds: float) -> None:
        captured_paths.append(destination)
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x00\x00" * 1600)

    monkeypatch.setattr(recognizer, "_capture", fake_capture)
    result = recognizer.listen(
        purpose="name", roster_hints=["Adam Wang", "Adam"], duration_seconds=4.5
    )

    assert result["transcripts"] == ["Adam Wang"]
    assert result["language"] == "en"
    assert model.calls[0][1]["hotwords"] == "Adam Wang Adam"
    assert model.calls[0][1]["language"] == "zh"
    assert isinstance(model.calls[0][0], np.ndarray)
    assert model.calls[0][0].dtype == np.float32
    assert captured_paths and not captured_paths[0].exists()


def test_confirmation_prompt_constrains_yes_and_no(monkeypatch):
    model = FakeModel()
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", model_factory=lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        recognizer,
        "_capture",
        lambda destination, duration: _write_test_wav(destination),
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


def _write_test_wav(destination: Path) -> None:
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)


def test_pcm_wav_loader_returns_normalized_float32(tmp_path):
    destination = tmp_path / "speech.wav"
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.array([-32768, 0, 32767], dtype="<i2").tobytes())

    samples = OfflineSpeechRecognizer._read_pcm_wav(destination)

    assert samples.dtype == np.float32
    assert samples.tolist() == [-1.0, 0.0, 32767 / 32768]


def test_uploaded_kebbi_audio_is_converted_transcribed_and_deleted(monkeypatch):
    model = FakeModel()
    recognizer = OfflineSpeechRecognizer(
        "ffmpeg.exe", model_factory=lambda *args, **kwargs: model
    )
    temporary_paths: list[Path] = []

    def fake_convert(source: Path, destination: Path) -> None:
        temporary_paths.extend([source, destination])
        assert source.suffix == ".ogg"
        assert source.read_bytes() == b"kebbi-opus-audio"
        _write_test_wav(destination)

    monkeypatch.setattr(recognizer, "_convert_uploaded_audio", fake_convert)
    result = recognizer.transcribe_upload(
        b"kebbi-opus-audio",
        content_type="audio/ogg; codecs=opus",
        purpose="name",
        roster_hints=["Adam Wang", "Adam"],
    )

    assert result["transcripts"] == ["Adam Wang"]
    assert temporary_paths and all(not path.exists() for path in temporary_paths)

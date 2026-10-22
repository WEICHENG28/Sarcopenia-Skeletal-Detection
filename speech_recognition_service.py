"""Offline speech recognition backed by the PC-connected C922 microphone.

The Kebbi Gecko browser does not expose the Web Speech API.  This module keeps
speech recognition on the analysis computer: FFmpeg captures a short WAV from
the C922 microphone and faster-whisper transcribes it locally.  Temporary audio
is always removed after transcription.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np


MICROPHONE_NAME = os.getenv("WEBCAM_MIC_NAME", "麥克風 (C922 Pro Stream Webcam)")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_ROBOT_NAME_PROMPTS = (
    "請完整",
    "請完整入鏡",
    "您好請清楚說出您的姓名",
)


class SpeechRecognitionError(RuntimeError):
    """A user-facing capture or transcription failure."""


class OfflineSpeechRecognizer:
    """Capture short utterances and transcribe them with a lazy-loaded model."""

    def __init__(
        self,
        ffmpeg: str,
        microphone_name: str = MICROPHONE_NAME,
        model_name: str = WHISPER_MODEL_NAME,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.microphone_name = microphone_name
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._listen_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        try:
            if self._model_factory is None:
                from faster_whisper import WhisperModel  # noqa: F401
            available = True
            message = "離線語音辨識可使用"
        except ImportError:
            available = False
            message = "尚未安裝 faster-whisper"
        return {
            "available": available,
            "model": self.model_name,
            "model_loaded": self.model_loaded,
            "microphone": self.microphone_name,
            "message": message,
        }

    def capture_command(self, destination: Path, duration_seconds: float) -> list[str]:
        return [
            self.ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow",
            "-i", f"audio={self.microphone_name}",
            "-t", f"{duration_seconds:.1f}",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(destination),
        ]

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise SpeechRecognitionError(
                        "離線語音辨識套件尚未安裝"
                    ) from exc
                factory = WhisperModel
            try:
                options = {
                    "device": "cpu",
                    "compute_type": "int8",
                    "cpu_threads": max(1, min(6, os.cpu_count() or 1)),
                    "num_workers": 1,
                }
                try:
                    # Prefer the existing cache so normal field use never waits on a network check.
                    self._model = factory(
                        self.model_name, local_files_only=True, **options
                    )
                except Exception:
                    # A clean installation downloads once; later launches use the branch above.
                    self._model = factory(
                        self.model_name, local_files_only=False, **options
                    )
            except Exception as exc:
                raise SpeechRecognitionError(
                    f"語音模型無法載入：{exc}"
                ) from exc
        return self._model

    def _capture(self, destination: Path, duration_seconds: float) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                self.capture_command(destination, duration_seconds),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=duration_seconds + 8,
                creationflags=creationflags,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechRecognitionError("C922 麥克風收音逾時") from exc
        except OSError as exc:
            raise SpeechRecognitionError(f"無法啟動 C922 麥克風：{exc}") from exc

        if completed.returncode != 0 or not destination.exists() or destination.stat().st_size < 1_000:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            if "already in use" in detail.lower() or "could not run graph" in detail.lower():
                detail = "C922 麥克風正被其他程式使用"
            raise SpeechRecognitionError(detail or "C922 麥克風沒有收到有效音訊")

    @staticmethod
    def _read_pcm_wav(audio_path: Path) -> np.ndarray:
        """Load the known FFmpeg PCM output without relying on PyAV.

        faster-whisper 1.2.1 passes ``metadata_errors`` to ``av.open`` while
        PyAV 19 removed that argument.  The capture format is already fixed to
        mono 16 kHz signed 16-bit PCM, so the standard-library WAV reader is
        both sufficient and insulated from that dependency mismatch.
        """
        try:
            with wave.open(str(audio_path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frames = source.readframes(source.getnframes())
        except (OSError, EOFError, wave.Error) as exc:
            raise SpeechRecognitionError(f"C922 錄音檔無法讀取：{exc}") from exc

        if (channels, sample_width, sample_rate) != (1, 2, 16000):
            raise SpeechRecognitionError(
                "C922 錄音格式不符，預期為單聲道、16 kHz、16-bit PCM"
            )
        if not frames:
            raise SpeechRecognitionError("C922 麥克風沒有收到有效音訊")
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def _is_robot_name_prompt(text: str) -> bool:
        """Recognize short echoes of Kebbi's own prompts, not user names."""
        normalized = re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text.casefold())
        return any(
            normalized == prompt
            or (3 <= len(normalized) <= len(prompt) and normalized in prompt)
            for prompt in _ROBOT_NAME_PROMPTS
        )

    def listen(
        self,
        *,
        purpose: str,
        roster_hints: list[str],
        duration_seconds: float = 4.5,
    ) -> dict[str, Any]:
        if not self._listen_lock.acquire(blocking=False):
            raise SpeechRecognitionError("另一段語音正在辨識，請稍候")

        descriptor, raw_path = tempfile.mkstemp(prefix="sarcopenia-speech-", suffix=".wav")
        os.close(descriptor)
        audio_path = Path(raw_path)
        try:
            self._capture(audio_path, duration_seconds)
            audio_samples = self._read_pcm_wav(audio_path)
            model = self._ensure_model()
            unique_hints = list(dict.fromkeys(
                item.strip() for item in roster_hints if item.strip()
            ))
            hints = "，".join(unique_hints)
            if purpose == "confirmation":
                initial_prompt = "回答是、對、正確、沒錯，或不是、不對。"
                hotwords = "是 對 正確 沒錯 不是 不對 yes no"
            else:
                initial_prompt = f"受測者會說出自己的姓名。名冊姓名可能是：{hints}。"
                hotwords = " ".join(unique_hints)

            try:
                segments, info = model.transcribe(
                    audio_samples,
                    task="transcribe",
                    language="zh",
                    beam_size=5,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords or None,
                )
                segment_texts = [segment.text.strip() for segment in segments if segment.text.strip()]
            except Exception as exc:
                raise SpeechRecognitionError(f"語音辨識失敗：{exc}") from exc

            if purpose == "name":
                segment_texts = [
                    text for text in segment_texts
                    if not self._is_robot_name_prompt(text)
                ]

            full_text = " ".join(segment_texts).strip()
            transcripts = list(dict.fromkeys(
                text for text in [full_text, *segment_texts] if text
            ))
            if not transcripts:
                if purpose == "name":
                    raise SpeechRecognitionError(
                        "只收到凱比提示音；請等提示結束後再說姓名"
                    )
                raise SpeechRecognitionError("沒有辨識到清楚語音")
            return {
                "transcripts": transcripts[:5],
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "model": self.model_name,
            }
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            finally:
                self._listen_lock.release()

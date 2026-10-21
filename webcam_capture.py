"""Shared C922 capture and server-side recording for the Kebbi web UI.

The browser running on Kebbi cannot open a webcam attached to the Windows PC.
This module keeps one FFmpeg capture process on the PC, exposes its latest JPEG
frame to FastAPI, and records selected intervals to MP4 on the PC.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator


CAMERA_NAME = os.getenv("WEBCAM_NAME", "c922 Pro Stream Webcam")


def find_ffmpeg() -> str:
    configured = os.getenv("FFMPEG_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path(shutil.which("ffmpeg")) if shutil.which("ffmpeg") else None,
        Path.home() / "anaconda3" / "envs" / "cv" / "Library" / "bin" / "ffmpeg.exe",
        Path.home() / "anaconda3" / "Library" / "bin" / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    raise RuntimeError("找不到 FFmpeg；請設定 FFMPEG_PATH。")


class CameraHub:
    """Capture one Windows webcam and share frames with preview and recorder."""

    def __init__(self, recordings_dir: Path, fps: int = 15) -> None:
        self.ffmpeg = find_ffmpeg()
        self.recordings_dir = recordings_dir
        self.fps = fps
        self.condition = threading.Condition()
        self.latest_frame: bytes | None = None
        self.sequence = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.recording_lock = threading.Lock()
        self.recording_process: subprocess.Popen[bytes] | None = None
        self.recording_id: str | None = None
        self.recording_path: Path | None = None
        self.recording_started_at: float | None = None

    def capture_command(self) -> list[str]:
        return [
            self.ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-f", "dshow",
            "-rtbufsize", "128M",
            "-video_size", "1280x720",
            "-framerate", "30",
            "-i", f"video={CAMERA_NAME}",
            "-an",
            "-vf", f"fps={self.fps},scale=960:-2",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "-f", "image2pipe",
            "pipe:1",
        ]

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="c922-camera-hub",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.stop_recording()
        except RuntimeError:
            pass
        process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        with self.condition:
            self.condition.notify_all()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            buffer = bytearray()
            process = subprocess.Popen(
                self.capture_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self.process = process
            stdout = process.stdout
            if stdout is None:
                process.terminate()
                time.sleep(1)
                continue

            try:
                while not self.stop_event.is_set():
                    chunk = stdout.read(64 * 1024)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            buffer.clear()
                            break
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start:
                                del buffer[:start]
                            break
                        frame = bytes(buffer[start : end + 2])
                        del buffer[: end + 2]
                        with self.condition:
                            self.latest_frame = frame
                            self.sequence += 1
                            self.condition.notify_all()
                        self._write_recording_frame(frame)
            finally:
                if process.poll() is None:
                    process.terminate()
                self.process = None

            if not self.stop_event.is_set():
                time.sleep(1)

    def get_frame(self, timeout: float = 5.0) -> bytes | None:
        self.start()
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest_frame is None and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return self.latest_frame

    def frames(self) -> Iterator[bytes]:
        self.start()
        last_sequence = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.sequence != last_sequence or self.stop_event.is_set(),
                    timeout=5,
                )
                if self.stop_event.is_set():
                    return
                frame = self.latest_frame
                last_sequence = self.sequence
            if frame is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

    def start_recording(self) -> str:
        if self.get_frame(timeout=3) is None:
            raise RuntimeError("C922 尚未產生畫面")

        with self.recording_lock:
            if self.recording_process and self.recording_process.poll() is None:
                raise RuntimeError("已有錄影正在進行")

            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            recording_id = uuid.uuid4().hex
            path = self.recordings_dir / f"{recording_id}.mp4"
            command = [
                self.ffmpeg,
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "image2pipe",
                "-framerate", str(self.fps),
                "-vcodec", "mjpeg",
                "-i", "pipe:0",
                "-an",
                # The bundled Conda FFmpeg has MPEG-4 Part 2 but no libx264.
                # MPEG-4 keeps the output in an MP4 container that OpenCV and
                # the existing skeleton-extraction pipeline can both read.
                "-c:v", "mpeg4",
                "-q:v", "4",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(path),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.recording_process = process
            self.recording_id = recording_id
            self.recording_path = path
            self.recording_started_at = time.monotonic()
            return recording_id

    def _write_recording_frame(self, frame: bytes) -> None:
        with self.recording_lock:
            process = self.recording_process
            if not process or process.poll() is not None or process.stdin is None:
                return
            try:
                process.stdin.write(frame)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                return

    def stop_recording(self) -> dict[str, object]:
        with self.recording_lock:
            process = self.recording_process
            recording_id = self.recording_id
            path = self.recording_path
            started_at = self.recording_started_at
            if not process or not recording_id or not path:
                raise RuntimeError("目前沒有正在進行的錄影")

            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

            stderr = b""
            if process.stderr:
                stderr = process.stderr.read()
                process.stderr.close()

            self.recording_process = None
            self.recording_id = None
            self.recording_path = None
            self.recording_started_at = None

            if process.returncode != 0 or not path.exists() or path.stat().st_size == 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"錄影轉檔失敗：{detail or 'FFmpeg 未產生影片'}")

            duration = time.monotonic() - started_at if started_at else 0.0
            return {
                "recording_id": recording_id,
                "path": path,
                "duration_seconds": round(duration, 1),
                "size_bytes": path.stat().st_size,
            }

    @property
    def is_recording(self) -> bool:
        with self.recording_lock:
            return bool(self.recording_process and self.recording_process.poll() is None)

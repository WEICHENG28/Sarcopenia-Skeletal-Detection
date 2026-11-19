"""Generate privacy-minimal QR cards for the automation test roster."""

from __future__ import annotations

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "automation-qr"
PARTICIPANT_IDS = ("SB01", "SB02", "SB03")


def generate() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = cv2.QRCodeEncoder_create()
    written: list[Path] = []
    for participant_id in PARTICIPANT_IDS:
        payload = f"SARCOPENIA:{participant_id}"
        qr = encoder.encode(payload)
        qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, (600, 600), interpolation=cv2.INTER_NEAREST)
        path = OUTPUT_DIR / f"{participant_id}.png"
        if not cv2.imwrite(str(path), qr):
            raise RuntimeError(f"無法寫入 {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    for output in generate():
        print(output)

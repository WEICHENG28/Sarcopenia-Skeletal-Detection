"""Mock tests for feature extraction pipeline.

Run:
    python test_features.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

from src.feature_extraction import extract_features


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_bmi_and_mot_with_trc() -> None:
    mot_path = Path("data/raw/mock_with_trc.mot")
    trc_path = Path("data/raw/mock_with_trc.trc")

    _write_text(
        mot_path,
        """# mock mot file
time hip_knee ankle
0.0 10 20
0.1 12 18
0.2 14 16
""",
    )
    _write_text(
        trc_path,
        """PathFileType 4 (X/Y/Z) sample.trc
DataRate CameraRate NumFrames NumMarkers Units
100.00 100.00 3 1 mm
Frame# Time M1X M1Y M1Z
1 0.00 1 2 3
2 0.01 2 3 4
3 0.02 3 4 5
""",
    )

    profile = {"age": 30, "height_cm": 175, "weight_kg": 70}
    feats = extract_features(profile=profile, mot_path=str(mot_path), trc_path=str(trc_path))

    expected_bmi = 70 / (1.75 * 1.75)
    assert abs(feats["bmi"] - expected_bmi) < 1e-9
    assert feats["mot_row_count"] == 3.0
    assert feats["trc_available"] == 1.0
    assert feats["trc_row_count"] == 3.0
    assert feats["mot_col_1_mean"] == 12.0


def test_optional_trc_missing_is_graceful() -> None:
    mot_path = Path("data/raw/mock_without_trc.mot")
    _write_text(
        mot_path,
        """time j1 j2
0.0 1 3
0.1 2 2
0.2 3 1
""",
    )

    profile = {"age": 22, "height_cm": 160, "weight_kg": 48}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = extract_features(profile=profile, mot_path=str(mot_path), trc_path=None)

    assert feats["trc_available"] == 0.0
    assert feats["trc_row_count"] == 0.0
    assert "TRC path not provided" in str(caught[-1].message)
    assert feats["mot_col_1_mean"] == 2.0


def run_all_tests() -> None:
    test_bmi_and_mot_with_trc()
    test_optional_trc_missing_is_graceful()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()


"""Minimal dimension checks for sequence feature extraction.

Run:
    python test_sequence_features.py
"""

from __future__ import annotations

from pathlib import Path

from src.feature_extraction import extract_sequence_features, pad_or_truncate_sequence


def _find_sample_mot() -> Path:
    candidates = sorted(Path("data/raw").glob("*.mot"))
    if not candidates:
        raise FileNotFoundError("No .mot file found under data/raw")
    return candidates[0]


def test_sequence_output_shape_and_static_bmi() -> None:
    mot_path = _find_sample_mot()
    trc_path = mot_path.with_suffix(".trc")
    trc_arg = str(trc_path) if trc_path.exists() else None

    profile = {
        "age": 30,
        "height_cm": 175,
        "weight_kg": 70,
        "gender": "male",
    }

    out = extract_sequence_features(
        profile=profile,
        mot_path=str(mot_path),
        trc_path=trc_arg,
        target_len=None,
        include_gender=True,
        trc_fill_dim=3,
    )

    seq = out["sequence"]
    assert isinstance(seq, list) and len(seq) > 0
    assert isinstance(seq[0], list) and len(seq[0]) > 0
    assert out["seq_len"] == len(seq)

    static = out["static"]
    static_names = out["static_feature_names"]
    assert "bmi" in static_names
    bmi_index = static_names.index("bmi")
    expected_bmi = 70 / (1.75 * 1.75)
    assert abs(static[bmi_index] - expected_bmi) < 1e-9


def test_padding_and_truncation_and_missing_trc_fallback() -> None:
    mot_path = _find_sample_mot()

    profile = {
        "age": 22,
        "height_cm": 160,
        "weight_kg": 48,
    }

    out = extract_sequence_features(
        profile=profile,
        mot_path=str(mot_path),
        trc_path=None,
        target_len=None,
        include_gender=False,
        trc_fill_dim=2,
    )

    seq = out["sequence"]
    seq_len = out["seq_len"]
    assert out["trc_available"] == 0.0
    assert seq_len == len(seq)

    padded = pad_or_truncate_sequence(seq, target_len=seq_len + 5)
    truncated = pad_or_truncate_sequence(seq, target_len=max(1, seq_len - 3))

    assert len(padded) == seq_len + 5
    assert len(truncated) == max(1, seq_len - 3)

    feature_dim = len(seq[0])
    assert all(len(row) == feature_dim for row in padded)
    assert all(len(row) == feature_dim for row in truncated)


def run_all_tests() -> None:
    test_sequence_output_shape_and_static_bmi()
    test_padding_and_truncation_and_missing_trc_fallback()
    print("Sequence feature tests passed.")


if __name__ == "__main__":
    run_all_tests()


"""Feature extraction entrypoint.

Combines:
1) Basic profile features (including BMI).
2) MOT-derived numeric summaries.
3) Optional TRC-derived numeric summaries with graceful fallback.

Also provides sequence-oriented extraction for DL pipelines:
- sequence: time_steps x features 2D list
- static: static feature vector (age/height/weight/BMI, optional gender encoding)
- seq_len: original sequence length before padding/truncation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, List

from .utils import extract_basic_features, parse_mot_features, parse_trc_features


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert value to float: {value!r}") from exc


def _iter_numeric_rows(lines: Iterable[str]) -> Iterable[List[float]]:
    """Yield whitespace/comma-separated numeric rows from text lines."""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        pieces = [p for p in line.replace(",", " ").split() if p]
        if not pieces:
            continue

        try:
            row = [_safe_float(p) for p in pieces]
        except ValueError:
            # Non-numeric header or metadata line; ignore.
            continue

        yield row


def _read_numeric_matrix(file_path: str) -> List[List[float]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    rows = list(_iter_numeric_rows(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"No numeric data found in file: {file_path}")

    return rows


def _pad_row(row: List[float], target_dim: int, pad_value: float = 0.0) -> List[float]:
    if len(row) >= target_dim:
        return row[:target_dim]
    return row + [pad_value] * (target_dim - len(row))


def _encode_gender(profile: Dict[str, Any]) -> float:
    """Encode gender to numeric when possible.

    Supported inputs:
    - numeric strings / numbers -> float value
    - common text labels: male/man/m -> 1.0, female/woman/f -> 0.0
    - otherwise -> 0.0
    """
    raw = profile.get("gender")
    if raw is None:
        return 0.0

    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip().lower()
    if text == "":
        return 0.0

    try:
        return float(text)
    except ValueError:
        pass

    if text in {"male", "man", "m", "boy"}:
        return 1.0
    if text in {"female", "woman", "f", "girl"}:
        return 0.0
    return 0.0


def pad_or_truncate_sequence(
    seq: List[List[float]],
    target_len: int,
    pad_value: float = 0.0,
    feature_dim: int | None = None,
) -> List[List[float]]:
    """Normalize sequence length to fixed target length.

    - If `len(seq) > target_len`: truncate from tail.
    - If `len(seq) < target_len`: right-pad with zero rows.
    """
    if target_len <= 0:
        raise ValueError("target_len must be > 0")

    if feature_dim is None:
        feature_dim = len(seq[0]) if seq else 0

    normalized_rows: List[List[float]] = []
    for row in seq:
        normalized_rows.append(_pad_row(list(row), target_dim=feature_dim, pad_value=pad_value))

    if len(normalized_rows) >= target_len:
        return normalized_rows[:target_len]

    missing = target_len - len(normalized_rows)
    pad_row = [pad_value] * feature_dim
    return normalized_rows + [pad_row[:] for _ in range(missing)]


def extract_sequence_features(
    profile: Dict[str, Any],
    mot_path: str,
    trc_path: str | None = None,
    target_len: int | None = None,
    include_gender: bool = True,
    trc_fill_dim: int = 0,
) -> Dict[str, Any]:
    """Extract sequence + static features for DL models.

    Args:
        profile: Basic personal metrics. Must include age, height_cm, weight_kg.
        mot_path: Required path to `.mot` file.
        trc_path: Optional path to `.trc` file.
        target_len: Optional fixed output sequence length.
        include_gender: Whether to append encoded gender to static vector.
        trc_fill_dim: Fallback TRC channel count when TRC is missing.

    Returns:
        Dict containing at least:
            - sequence: 2D list of shape (time_steps, feature_dim)
            - static: static feature vector [age, height_cm, weight_kg, bmi, (gender)]
            - seq_len: original MOT sequence length before normalization
    """
    if trc_fill_dim < 0:
        raise ValueError("trc_fill_dim must be >= 0")

    basic = extract_basic_features(profile)
    static_feature_names = ["age", "height_cm", "weight_kg", "bmi"]
    static = [
        float(basic["age"]),
        float(basic["height_cm"]),
        float(basic["weight_kg"]),
        float(basic["bmi"]),
    ]
    if include_gender:
        static.append(_encode_gender(profile))
        static_feature_names.append("gender")

    mot_rows = _read_numeric_matrix(mot_path)
    mot_dim = max(len(row) for row in mot_rows)
    mot_matrix = [_pad_row(row, mot_dim, pad_value=0.0) for row in mot_rows]

    trc_available = 0.0
    trc_matrix: List[List[float]] = []
    trc_dim = int(trc_fill_dim)
    if trc_path:
        trc_file = Path(trc_path)
        if trc_file.exists():
            parsed_trc = _read_numeric_matrix(str(trc_file))
            trc_dim = max(len(row) for row in parsed_trc)
            trc_matrix = [_pad_row(row, trc_dim, pad_value=0.0) for row in parsed_trc]
            trc_available = 1.0

    seq_len = len(mot_matrix)
    sequence: List[List[float]] = []
    for idx in range(seq_len):
        row = list(mot_matrix[idx])
        if trc_dim > 0:
            trc_row = trc_matrix[idx] if idx < len(trc_matrix) else [0.0] * trc_dim
            row.extend(trc_row)
        sequence.append(row)

    feature_dim = len(sequence[0]) if sequence else mot_dim + trc_dim
    if target_len is not None:
        sequence = pad_or_truncate_sequence(
            sequence,
            target_len=target_len,
            pad_value=0.0,
            feature_dim=feature_dim,
        )

    return {
        "sequence": sequence,
        "static": static,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "static_feature_names": static_feature_names,
        "trc_available": trc_available,
    }


def extract_sequence_features_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper for sequence extraction from record schema."""
    if "profile" not in record:
        raise KeyError("record missing required key: profile")
    if "mot_path" not in record:
        raise KeyError("record missing required key: mot_path")

    return extract_sequence_features(
        profile=record["profile"],
        mot_path=str(record["mot_path"]),
        trc_path=record.get("trc_path"),
        target_len=record.get("target_len"),
    )


def extract_features(
    profile: Dict[str, float],
    mot_path: str,
    trc_path: str | None = None,
) -> Dict[str, float]:
    """Extract all available features for one sample.

    Args:
        profile: Basic personal metrics. Must include age, height_cm, weight_kg.
        mot_path: Required path to `.mot` file.
        trc_path: Optional path to `.trc` file.

    Returns:
        Flat numeric feature dictionary.
    """
    features: Dict[str, float] = {}
    features.update(extract_basic_features(profile))
    features.update(parse_mot_features(mot_path))
    features.update(parse_trc_features(trc_path))
    return features


def extract_features_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    """Convenience wrapper for a common record schema.

    Expected record keys:
        - profile: dict with age/height_cm/weight_kg
        - mot_path: str
        - trc_path: optional str
    """
    if "profile" not in record:
        raise KeyError("record missing required key: profile")
    if "mot_path" not in record:
        raise KeyError("record missing required key: mot_path")

    return extract_features(
        profile=record["profile"],
        mot_path=str(record["mot_path"]),
        trc_path=record.get("trc_path"),
    )


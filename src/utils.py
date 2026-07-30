"""Utility functions for feature extraction pipeline.

This module provides:
1) Basic demographic parsing and BMI calculation.
2) `.mot` parsing for joint/time-series numeric features.
3) Optional `.trc` parsing with graceful fallback behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math
import warnings


def _safe_float(value: str) -> float:
    """Convert a string to float with strict error reporting."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert value to float: {value!r}") from exc


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    """Compute BMI from height (cm) and weight (kg).

    Raises:
        ValueError: if height or weight are non-positive.
    """
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")

    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def extract_basic_features(profile: Dict[str, float]) -> Dict[str, float]:
    """Extract basic profile features and BMI.

    Expected keys:
        - age
        - height_cm
        - weight_kg
    """
    required = ("age", "height_cm", "weight_kg")
    missing = [k for k in required if k not in profile]
    if missing:
        raise KeyError(f"Missing required profile keys: {missing}")

    age = float(profile["age"])
    height_cm = float(profile["height_cm"])
    weight_kg = float(profile["weight_kg"])

    return {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": compute_bmi(height_cm=height_cm, weight_kg=weight_kg),
    }


def _iter_numeric_rows(lines: Iterable[str]) -> Iterable[List[float]]:
    """Yield whitespace/comma separated numeric rows from text lines."""
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


def _series_summary(values: List[float], prefix: str) -> Dict[str, float]:
    """Create summary statistics for a numeric series."""
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
        }

    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_std": math.sqrt(variance),
    }


def parse_mot_features(mot_path: str) -> Dict[str, float]:
    """Parse `.mot` file and return robust joint/time features.

    Behavior:
    - If file is missing: raise FileNotFoundError.
    - Ignore non-numeric lines safely.
    - Summarize each numeric column as col_{i}_{mean/min/max/std}.
    """
    path = Path(mot_path)
    if not path.exists():
        raise FileNotFoundError(f"MOT file does not exist: {mot_path}")

    rows = list(_iter_numeric_rows(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"No numeric data found in MOT file: {mot_path}")

    max_cols = max(len(r) for r in rows)
    features: Dict[str, float] = {"mot_row_count": float(len(rows))}

    for idx in range(max_cols):
        col_values = [r[idx] for r in rows if len(r) > idx]
        features.update(_series_summary(col_values, prefix=f"mot_col_{idx}"))

    return features


def parse_trc_features(trc_path: str | None) -> Dict[str, float]:
    """Parse optional `.trc` file.

    Graceful behavior:
    - If trc_path is None or empty: return defaults with availability flag.
    - If file does not exist: warn and return defaults.
    - If file exists but has no numeric rows: warn and return defaults.
    """
    defaults = {
        "trc_available": 0.0,
        "trc_row_count": 0.0,
        "trc_col_0_mean": 0.0,
        "trc_col_0_min": 0.0,
        "trc_col_0_max": 0.0,
        "trc_col_0_std": 0.0,
    }

    if not trc_path:
        warnings.warn("TRC path not provided; using default TRC features.", RuntimeWarning)
        return defaults

    path = Path(trc_path)
    if not path.exists():
        warnings.warn(f"TRC file not found: {trc_path}; using defaults.", RuntimeWarning)
        return defaults

    rows = list(_iter_numeric_rows(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        warnings.warn(
            f"TRC file has no numeric rows: {trc_path}; using defaults.",
            RuntimeWarning,
        )
        return defaults

    max_cols = max(len(r) for r in rows)
    features: Dict[str, float] = {
        "trc_available": 1.0,
        "trc_row_count": float(len(rows)),
    }
    for idx in range(max_cols):
        col_values = [r[idx] for r in rows if len(r) > idx]
        features.update(_series_summary(col_values, prefix=f"trc_col_{idx}"))

    return features


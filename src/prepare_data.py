"""Prepare long-format action-sample dataset for sequence model training.

Flow:
1) Read ``data/raw_registry.csv``.
2) For each user row and each action (Sit/Walk/Squat), build one sample row.
3) Use Step 2 sequence interface from ``src.feature_extraction``.
4) Normalize sequence length via padding/truncation to fixed target length.
5) Write long-format matrix to ``data/database.csv``.

Output row schema (minimum):
- User_ID, Label, Action, ActionFlag
- seq_len
- static_* (including static_bmi)
- seq_t{time}_f{feature} flattened sequence columns
"""

from __future__ import annotations

import csv
import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import extract_sequence_features


RAW_REGISTRY_PATH = PROJECT_ROOT / "data" / "raw_registry.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "database.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

ACTION_SPECS = {
    "Sit": ("Sit_mot_path", "Sit_trc_path"),
    "Walk": ("Walk_mot_path", "Walk_trc_path"),
    "Squat": ("Squat_mot_path", "Squat_trc_path"),
}

DEFAULT_TARGET_LEN = 128


def _as_stripped(value: str | None) -> str:
    return (value or "").strip()


def _as_float_or_default(value: str | None, default: float = 0.0) -> float:
    raw = _as_stripped(value)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"Numeric parse failed for value {value!r}; use default {default}.", RuntimeWarning)
        return default


def _parse_binary_label(value: str | None) -> int | None:
    """Parse label into binary int (0/1); return None for unsupported values."""
    raw = _as_stripped(value)
    if raw == "":
        return None
    try:
        parsed = int(float(raw))
    except ValueError:
        return None
    return parsed if parsed in (0, 1) else None


def _flatten_sequence(sequence: List[List[float]], target_len: int, feature_dim: int) -> Dict[str, str]:
    """Flatten [T, F] sequence into stable column naming: seq_t000_f000..."""
    out: Dict[str, str] = {}
    for t in range(target_len):
        row = sequence[t] if t < len(sequence) else []
        for f in range(feature_dim):
            v = row[f] if f < len(row) else 0.0
            out[f"seq_t{t:03d}_f{f:03d}"] = str(float(v))
    return out


def _resolve_io_path(path_str: str) -> str:
    """Resolve relative path against project root for stable CLI execution cwd."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _find_bridge_candidate(user_id: str, action: str) -> Tuple[str, str]:
    """Infer bridge output path by `[User_ID]_[Action].mot/.trc` with tiny fallback."""
    if user_id == "":
        return "", ""

    # Primary rule: [ID]_[Action].mot/.trc
    action_tokens = [action, action.lower(), action.upper(), action.capitalize()]
    checked: set[str] = set()
    for token in action_tokens:
        if token in checked:
            continue
        checked.add(token)
        stem = f"{user_id}_{token}"
        mot = RAW_DATA_DIR / f"{stem}.mot"
        if mot.exists():
            trc = RAW_DATA_DIR / f"{stem}.trc"
            return str(mot), str(trc)

    # Tiny fallback for mock/single-action bridge outputs: [ID]_*.mot (only when unique)
    mot_candidates = sorted(RAW_DATA_DIR.glob(f"{user_id}_*.mot"))
    if len(mot_candidates) == 1:
        mot = mot_candidates[0]
        trc = mot.with_suffix(".trc")
        return str(mot), str(trc)

    return "", ""


def _resolve_action_paths(
    *,
    raw: Dict[str, str],
    action: str,
    mot_col: str,
    trc_col: str,
    auto_bridge_path: bool,
) -> Tuple[str, str]:
    """Resolve MOT/TRC path with backward-compatible registry-first strategy."""
    mot_path = _as_stripped(raw.get(mot_col))
    trc_path = _as_stripped(raw.get(trc_col))

    # Legacy path mode keeps highest priority.
    if mot_path != "":
        mot_resolved = _resolve_io_path(mot_path)
        trc_resolved = _resolve_io_path(trc_path) if trc_path != "" else ""
        return mot_resolved, trc_resolved

    if not auto_bridge_path:
        return "", ""

    user_id = _as_stripped(raw.get("User_ID"))
    return _find_bridge_candidate(user_id=user_id, action=action)


def _build_action_row(
    *,
    raw: Dict[str, str],
    idx: int,
    action: str,
    mot_col: str,
    trc_col: str,
    target_len: int,
    auto_bridge_path: bool,
) -> Dict[str, str]:
    """Build one long-format action row with error tolerance."""
    user_id = _as_stripped(raw.get("User_ID"))
    label_raw = _as_stripped(raw.get("Label"))
    parsed_label = _parse_binary_label(label_raw)
    profile: Dict[str, Any] = {
        "age": _as_float_or_default(raw.get("age"), 0.0),
        "height_cm": _as_float_or_default(raw.get("height_cm"), 0.0),
        "weight_kg": _as_float_or_default(raw.get("weight_kg"), 0.0),
        "gender": _as_stripped(raw.get("gender")),
    }

    out: Dict[str, str] = {
        "User_ID": user_id,
        "Label": str(parsed_label) if parsed_label is not None else label_raw,
        "Action": action,
        "ActionFlag": "0",
        "seq_len": "0",
    }

    mot_path, trc_path = _resolve_action_paths(
        raw=raw,
        action=action,
        mot_col=mot_col,
        trc_col=trc_col,
        auto_bridge_path=auto_bridge_path,
    )
    if mot_path == "":
        # Missing MOT: keep row alive and mark unavailable.
        return out

    if not Path(mot_path).exists():
        # Broken path or unresolved bridge output: keep row alive.
        return out

    try:
        seq_pack = extract_sequence_features(
            profile=profile,
            mot_path=mot_path,
            trc_path=trc_path or None,
            target_len=target_len,
            include_gender=True,
        )
    except Exception as exc:
        warnings.warn(
            f"Row {idx} ({user_id or 'UNKNOWN'}), action {action}: "
            f"sequence extraction failed ({exc}); set ActionFlag=0 and continue.",
            RuntimeWarning,
        )
        return out

    out["ActionFlag"] = "1"
    out["seq_len"] = str(int(seq_pack.get("seq_len", 0)))

    static_names = list(seq_pack.get("static_feature_names", []))
    static_values = list(seq_pack.get("static", []))
    for name, value in zip(static_names, static_values):
        out[f"static_{name}"] = str(float(value))

    sequence = seq_pack.get("sequence", [])
    feature_dim = int(seq_pack.get("feature_dim", 0))
    out.update(_flatten_sequence(sequence, target_len=target_len, feature_dim=feature_dim))
    return out


def build_rows(raw_rows: List[Dict[str, str]], target_len: int) -> List[Dict[str, str]]:
    """Convert registry into long-format rows: one row per user-action sample."""
    prepared_rows: List[Dict[str, str]] = []
    for idx, raw in enumerate(raw_rows, start=2):
        for action, (mot_col, trc_col) in ACTION_SPECS.items():
            prepared_rows.append(
                _build_action_row(
                    raw=raw,
                    idx=idx,
                    action=action,
                    mot_col=mot_col,
                    trc_col=trc_col,
                    target_len=target_len,
                    auto_bridge_path=False,
                )
            )
    return prepared_rows


def build_rows_with_mode(
    raw_rows: List[Dict[str, str]],
    target_len: int,
    auto_bridge_path: bool,
) -> List[Dict[str, str]]:
    """Convert registry into long-format rows with optional bridge path inference."""
    prepared_rows: List[Dict[str, str]] = []
    for idx, raw in enumerate(raw_rows, start=2):
        for action, (mot_col, trc_col) in ACTION_SPECS.items():
            prepared_rows.append(
                _build_action_row(
                    raw=raw,
                    idx=idx,
                    action=action,
                    mot_col=mot_col,
                    trc_col=trc_col,
                    target_len=target_len,
                    auto_bridge_path=auto_bridge_path,
                )
            )
    return prepared_rows


def _infer_feature_dim(rows: List[Dict[str, str]]) -> int:
    """Infer global flattened sequence feature dimension from generated rows."""
    max_feature_idx = -1
    for row in rows:
        for key in row.keys():
            if not key.startswith("seq_t"):
                continue
            try:
                f_idx = int(key.rsplit("_f", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue
            if f_idx > max_feature_idx:
                max_feature_idx = f_idx
    return max_feature_idx + 1


def write_matrix(rows: List[Dict[str, str]], output_path: Path, target_len: int) -> None:
    """Write long-format rows with stable mandatory/static/sequence column ordering."""
    if not rows:
        raise ValueError("No rows generated from raw registry.")

    fixed_head = ["User_ID", "Label", "Action", "ActionFlag", "seq_len"]
    static_cols = sorted({k for row in rows for k in row.keys() if k.startswith("static_")})

    feature_dim = _infer_feature_dim(rows)
    seq_cols = [f"seq_t{t:03d}_f{f:03d}" for t in range(target_len) for f in range(feature_dim)]
    columns = fixed_head + static_cols + seq_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for col in columns:
                value = row.get(col, "")
                if value != "":
                    normalized[col] = value
                elif col in ("User_ID", "Label", "Action"):
                    normalized[col] = ""
                elif col in ("ActionFlag", "seq_len"):
                    normalized[col] = "0"
                else:
                    normalized[col] = "0.0"
            writer.writerow(normalized)


def print_preparation_summary(rows: List[Dict[str, str]]) -> None:
    """Print minimal summary for sanity check."""
    total = len(rows)
    flagged = sum(1 for r in rows if (r.get("ActionFlag") or "0") == "1")
    print(f"Prepared action samples: {total}")
    print(f"ActionFlag=1 samples: {flagged}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare long-format sequence dataset")
    parser.add_argument(
        "--raw-registry",
        type=Path,
        default=RAW_REGISTRY_PATH,
        help=f"Path to raw registry CSV (default: {RAW_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output long-format CSV path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=DEFAULT_TARGET_LEN,
        help=f"Fixed sequence length after pad/truncate (default: {DEFAULT_TARGET_LEN})",
    )
    parser.add_argument(
        "--bridge-mode",
        action="store_true",
        help="Enable auto bridge path inference: data/raw/[User_ID]_[Action].mot/.trc when explicit paths are missing.",
    )
    parser.add_argument(
        "--auto-bridge-path",
        action="store_true",
        help="Alias of --bridge-mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_len <= 0:
        raise ValueError("target_len must be > 0")

    raw_registry_path = Path(args.raw_registry)
    output_path = Path(args.output)
    auto_bridge_path = bool(args.bridge_mode or args.auto_bridge_path)

    if not raw_registry_path.exists():
        raise FileNotFoundError(f"Raw registry not found: {raw_registry_path}")

    with raw_registry_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("raw_registry.csv has no header row")

        required_base = [
            "User_ID",
            "Label",
            "age",
            "gender",
            "height_cm",
            "weight_kg",
        ]
        required_paths = [
            "Sit_mot_path",
            "Sit_trc_path",
            "Walk_mot_path",
            "Walk_trc_path",
            "Squat_mot_path",
            "Squat_trc_path",
        ]
        required = required_base if auto_bridge_path else (required_base + required_paths)
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"raw_registry.csv missing required columns: {missing}")

        raw_rows = list(reader)

    prepared = build_rows_with_mode(
        raw_rows,
        target_len=args.target_len,
        auto_bridge_path=auto_bridge_path,
    )
    write_matrix(prepared, output_path, target_len=args.target_len)
    print_preparation_summary(prepared)
    print(f"target_len: {args.target_len}")
    print(f"bridge_mode: {int(auto_bridge_path)}")
    print(f"Wrote training matrix: {output_path}")


if __name__ == "__main__":
    main()


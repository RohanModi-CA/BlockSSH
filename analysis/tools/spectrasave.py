from __future__ import annotations

from pathlib import Path
from typing import Any

import msgpack
import numpy as np


AUTO_SPECTRASAVE = "__auto__"
DEFAULT_SPECTRASAVE_DIR = Path(__file__).resolve().parents[1] / "spectrasave"


def get_default_spectrasave_dir() -> Path:
    return DEFAULT_SPECTRASAVE_DIR


def add_spectrasave_arg(parser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--spectrasave",
        nargs="?",
        const=AUTO_SPECTRASAVE,
        default=None,
        metavar="PATH",
        help=help_text
        or (
            "Save the produced 1D spectrum as msgpack. Pass no value to use an auto-generated "
            "path under spectrasave/."
        ),
    )


def _sanitize_token(value: Any) -> str:
    text = str(value).strip().lower()
    out_chars: list[str] = []
    last_was_sep = False
    for ch in text:
        if ch.isalnum():
            out_chars.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                out_chars.append("-")
                last_was_sep = True
    sanitized = "".join(out_chars).strip("-")
    return sanitized or "unnamed"


def build_default_spectrasave_name(*parts: Any) -> str:
    tokens = [_sanitize_token(part) for part in parts if str(part).strip()]
    if not tokens:
        tokens = ["spectrum"]
    return "__".join(tokens) + ".msgpack"


def resolve_spectrasave_path(
    requested: str | Path | None,
    *,
    default_name: str,
    multi_suffix: str | None = None,
) -> Path | None:
    if requested is None:
        return None

    if str(requested) == AUTO_SPECTRASAVE:
        path = get_default_spectrasave_dir() / default_name
    else:
        raw_path = Path(requested)
        if raw_path.suffix:
            path = raw_path
            if multi_suffix is not None:
                path = raw_path.with_name(f"{raw_path.stem}__{_sanitize_token(multi_suffix)}{raw_path.suffix}")
        else:
            path = raw_path / default_name

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_msgpackable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_msgpackable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_msgpackable(v) for v in value]
    return value


def save_spectrum_msgpack(
    path: str | Path,
    *,
    freq: np.ndarray,
    amplitude: np.ndarray,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    freq_arr = np.asarray(freq, dtype=float)
    amp_arr = np.asarray(amplitude, dtype=float)
    if freq_arr.ndim != 1 or amp_arr.ndim != 1 or freq_arr.size != amp_arr.size:
        raise ValueError("freq and amplitude must be 1D arrays of equal length")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifactType": "spectrum",
        "artifactVersion": 1,
        "label": None if label is None else str(label),
        "freq": freq_arr.tolist(),
        "amplitude": amp_arr.tolist(),
        "metadata": _to_msgpackable(metadata or {}),
    }
    with open(output, "wb") as f:
        packed = msgpack.packb(payload, use_bin_type=True)
        f.write(packed)
    return output


def load_spectrum_msgpack(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return {
        "artifactType": data["artifactType"],
        "artifactVersion": data["artifactVersion"],
        "label": data.get("label"),
        "freq": np.asarray(data["freq"], dtype=float),
        "amplitude": np.asarray(data["amplitude"], dtype=float),
        "metadata": data.get("metadata", {}),
    }

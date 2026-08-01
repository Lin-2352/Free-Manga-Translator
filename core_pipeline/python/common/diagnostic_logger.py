from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_paths import PROJECT_ROOT


_LOCK = threading.Lock()
_REDACT_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "apiKey",
    "base64Data",
    "bearer",
    "imageData",
    "imageDataUrl",
    "key",
    "secret",
    "token",
    "translatedImageDataUrl",
}
# "sk-" (OpenAI/OpenRouter legacy), "AIza" (Google/Gemini), "gsk_" (Groq), "sk-or-" (OpenRouter),
# "csk-" (Cerebras), "fw_" (Fireworks), "nvapi-" (NVIDIA) are documented key-prefix conventions
# for those providers, so a bare substring match is safe. Mistral and Cloudflare do not publish a
# distinguishing key prefix (their keys/tokens are opaque hex/alphanumeric strings indistinguishable
# from other data) -- named-field redaction via _REDACT_KEYS is the only defense for those two.
_REDACT_VALUE_MARKERS = (
    "sk-",
    "AIza",
    "gsk_",
    "Bearer ",
    "sk-or-",
    "csk-",
    "fw_",
    "nvapi-",
)


def diagnostics_enabled() -> bool:
    return os.environ.get("FMT_DIAGNOSTICS_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}


def diagnostics_dir() -> Path:
    configured = os.environ.get("FMT_DIAGNOSTICS_DIR", "").strip()
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "runtime_logs"


def diagnostics_log_path(day: datetime | None = None) -> Path:
    stamp = (day or datetime.now()).strftime("%Y%m%d")
    return diagnostics_dir() / f"diagnostics_{stamp}.jsonl"


# Log files already rotate by day (via the date-stamped filename above), but nothing bounded
# an individual day's file size -- a burst-heavy day could grow the file unbounded. This is a
# simple size-based rotation: once a day's file exceeds the threshold, it is renamed aside
# (keeping up to _ROTATION_BACKUPS numbered backups) and a fresh file is started, matching the
# spirit of logging.handlers.RotatingFileHandler without requiring this module to hold onto a
# long-lived handler object across day boundaries (the path is recomputed per-write already).
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_ROTATION_BACKUPS = 3


def _rotation_max_bytes() -> int:
    value = os.environ.get("FMT_DIAGNOSTICS_MAX_BYTES", "").strip()
    try:
        parsed = int(value) if value else _DEFAULT_MAX_BYTES
    except ValueError:
        parsed = _DEFAULT_MAX_BYTES
    return max(0, parsed)


def _rotate_if_needed(path: Path) -> None:
    max_bytes = _rotation_max_bytes()
    if max_bytes <= 0:
        return
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    for index in range(_ROTATION_BACKUPS - 1, 0, -1):
        older = path.with_name(f"{path.name}.{index}")
        newer = path.with_name(f"{path.name}.{index + 1}")
        if older.exists():
            try:
                older.replace(newer)
            except OSError:
                pass
    backup_one = path.with_name(f"{path.name}.1")
    try:
        path.replace(backup_one)
    except OSError:
        pass


def _safe_text(value: str, max_len: int = 900) -> str:
    cleaned = value.replace("\r", "\\r").replace("\n", "\\n")
    if any(marker in cleaned for marker in _REDACT_VALUE_MARKERS):
        return "<redacted>"
    return cleaned[: max(0, max_len - 3)] + "..." if len(cleaned) > max_len else cleaned


def _safe_payload(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "<max-depth>"
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            key_folded = key.replace("_", "").replace("-", "").lower()
            if key in _REDACT_KEYS or key_folded in {item.lower() for item in _REDACT_KEYS}:
                safe[key] = "<redacted>"
            else:
                safe[key] = _safe_payload(raw_value, depth + 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item, depth + 1) for item in value[:50]]
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_text(str(value), max_len=400)


def write_diagnostic_event(
    event: str,
    details: dict[str, Any] | None = None,
    *,
    trace_id: str | None = None,
    source: str = "backend",
    level: str = "info",
) -> Path | None:
    if not diagnostics_enabled():
        return None
    now = datetime.now(timezone.utc)
    record = {
        "timestamp": now.isoformat(),
        "level": level,
        "source": source,
        "traceId": trace_id or "no-trace",
        "event": event,
        "details": _safe_payload(details or {}),
    }
    path = diagnostics_log_path(now)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path

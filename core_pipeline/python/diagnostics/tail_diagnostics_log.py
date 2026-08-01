from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for rel in ("python/common",):
    path = str(PROJECT_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from diagnostic_logger import diagnostics_log_path


def _format_record(record: dict, raw: bool = False) -> str:
    if raw:
        return json.dumps(record, ensure_ascii=False)
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    trace_id = record.get("traceId") or "no-trace"
    level = str(record.get("level") or "info").upper()
    event = record.get("event") or "diagnostic"
    source = record.get("source") or "backend"
    timestamp = record.get("timestamp") or ""
    fragments = []
    for key in ("status", "reason", "error", "provider", "sample", "kept", "rejected", "renderedRegions", "translationCount"):
        if key in details and details[key] not in ("", None):
            fragments.append(f"{key}={details[key]}")
    suffix = " ".join(fragments)
    return f"{timestamp} [{level}] {source} trace={trace_id} event={event}" + (f" {suffix}" if suffix else "")


def _read_new_records(path: Path, offset: int, raw: bool = False) -> int:
    if not path.exists():
        return offset
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(line, flush=True)
                continue
            print(_format_record(record, raw=raw), flush=True)
        return handle.tell()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail the structured runtime diagnostics log.")
    parser.add_argument("--path", type=Path, default=None, help="Explicit diagnostics JSONL path.")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON records.")
    parser.add_argument("--last", type=int, default=40, help="Initial number of records to print.")
    parser.add_argument("--follow", action="store_true", help="Keep waiting for new records.")
    parser.add_argument("--poll", type=float, default=0.5, help="Polling interval in seconds while following.")
    args = parser.parse_args()

    path = args.path or diagnostics_log_path(datetime.now())
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-max(0, args.last):]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            print(_format_record(record, raw=args.raw), flush=True)
        except json.JSONDecodeError:
            print(line, flush=True)

    offset = path.stat().st_size
    if not args.follow:
        return 0

    print(f"[tail-diagnostics] following {path}", flush=True)
    while True:
        offset = _read_new_records(path, offset, raw=args.raw)
        time.sleep(max(0.1, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())

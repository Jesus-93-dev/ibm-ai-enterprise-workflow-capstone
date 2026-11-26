import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_event(
    log_path: str | Path,
    event_type: str,
    payload: dict[str, Any],
    novelty_flag: bool = False,
    metric_value: float | None = None,
) -> dict[str, Any]:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "novelty_flag": novelty_flag,
        "metric_value": metric_value,
        "payload": payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_logs(log_path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]

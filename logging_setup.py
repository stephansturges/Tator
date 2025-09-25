import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Merge any structured extras
        for key, value in getattr(record, "__dict__", {}).items():
            if key not in payload and key not in ("args", "msg"):
                # Only include simple JSON-serializable values
                try:
                    json.dumps(value)
                    payload[key] = value
                except Exception:
                    pass
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, json_logs: bool = True) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


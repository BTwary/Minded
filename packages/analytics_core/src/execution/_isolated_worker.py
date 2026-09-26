"""Worker entrypoint for independently executed isolated calls using secure Arrow/JSON IPC."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from packages.analytics_core.src.execution.process_isolation import (
    deserialize_ipc,
    resolve_callable,
    serialize_ipc,
)


def main(request_path: str, response_path: str) -> None:
    req_file = Path(request_path)
    resp_file = Path(response_path)
    dir_path = req_file.parent

    try:
        req_data = json.loads(req_file.read_text(encoding="utf-8"))
        fn = resolve_callable(req_data["module"], req_data["qualname"])
        args = deserialize_ipc(req_data.get("args", []), dir_path)
        kwargs = deserialize_ipc(req_data.get("kwargs", {}), dir_path)
        result = fn(*args, **kwargs)
        serialized_result = serialize_ipc(result, dir_path, "res")
        payload = {"ok": True, "result": serialized_result, "error": None}
    except Exception as exc:
        payload = {"ok": False, "result": None, "error": repr(exc)}

    resp_file.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


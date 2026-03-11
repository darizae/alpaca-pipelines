from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.fs import FileSystem


def write_json(path: Path, payload: Any, fs: FileSystem) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    fs.write_text(path, content)


def read_json(path: Path, fs: FileSystem) -> Any:
    content = fs.read_text(path)
    data: Any = json.loads(content)
    return data

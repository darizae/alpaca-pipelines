from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem


def write_json(path: Path, payload: Any, fs: FileSystem = _DEFAULT_FS) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    fs.write_text(path, content)


def read_json(path: Path, fs: FileSystem = _DEFAULT_FS) -> Any:
    content = fs.read_text(path)
    data: Any = json.loads(content)
    return data


def write_csv_rows(path: Path, rows: list[list[str]], fs: FileSystem = _DEFAULT_FS) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    fs.write_text(path, buffer.getvalue())


def read_csv_rows(path: Path, fs: FileSystem = _DEFAULT_FS) -> list[list[str]]:
    content = fs.read_text(path)
    buffer = io.StringIO(content)
    reader = csv.reader(buffer)
    return list(reader)

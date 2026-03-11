from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class FileSystem(Protocol):
    """
    Minimal filesystem interface used by all library workflows.

    LocalFS (below) implements this against pathlib.Path.
    alpaca-ui provides a concrete remote implementation (e.g., via SSH/SFTP).
    """

    def exists(self, path: Path) -> bool: ...

    def is_dir(self, path: Path) -> bool: ...

    def is_file(self, path: Path) -> bool: ...

    def iterdir(self, path: Path) -> list[Path]: ...

    def rglob_wavs(self, path: Path) -> list[Path]: ...

    def read_text(self, path: Path, encoding: str = "utf-8") -> str: ...

    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None: ...

    def open_read(self, path: Path) -> BinaryIO:
        """
        Open file for binary reading.

        The returned object MUST support seek() so that soundfile can read
        WAV headers without consuming the entire stream.
        Caller is responsible for closing.
        """
        ...

    def open_write(self, path: Path) -> BinaryIO:
        """
        Open file for binary writing (truncate/overwrite).

        Caller is responsible for closing.
        """
        ...

    def makedirs(self, path: Path) -> None: ...

    def rename(self, src: Path, dst: Path) -> None:
        """
        Rename src to dst.

        MUST NOT overwrite dst if it exists — raise FileExistsError if so.
        Caller guarantees dst.parent exists before calling.
        """
        ...

    def unlink(self, path: Path) -> None: ...


class LocalFS:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def iterdir(self, path: Path) -> list[Path]:
        return list(path.iterdir())

    def rglob_wavs(self, path: Path) -> list[Path]:
        return sorted(p for p in path.rglob("*.wav") if p.is_file())

    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        return path.read_text(encoding=encoding)

    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    def open_read(self, path: Path) -> BinaryIO:
        return open(path, "rb")

    def open_write(self, path: Path) -> BinaryIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "wb")

    def makedirs(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def rename(self, src: Path, dst: Path) -> None:
        if dst.exists():
            raise FileExistsError(f"Target exists, refusing to overwrite: {dst}")
        src.rename(dst)

    def unlink(self, path: Path) -> None:
        path.unlink()


_DEFAULT_FS: FileSystem = LocalFS()

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Rollback types
# ---------------------------------------------------------------------------


@dataclass
class RollbackArtifact:
    """Describes the partial state left after an incomplete rollback."""

    completed_moves: list[tuple[str, str]] = field(default_factory=list)
    """(src, dst) pairs for rename ops that completed Phase 2 successfully."""

    pending_temps: list[tuple[str, str]] = field(default_factory=list)
    """(temp_path, original_src) for temps that could not be restored."""

    rollback_errors: list[str] = field(default_factory=list)
    """One error message per failed rollback rename."""


class RollbackIncompleteError(RuntimeError):
    """
    Raised when a rename plan failed mid-apply and rollback was also incomplete.

    The ``artifact`` attribute describes what moved and what is stuck in temp state.
    Callers should serialize the artifact and store it for manual recovery.
    """

    def __init__(self, message: str, artifact: RollbackArtifact) -> None:
        super().__init__(message)
        self.artifact = artifact


# ---------------------------------------------------------------------------
# FileSystem protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FileSystem(Protocol):
    """
    Minimal filesystem interface used by all library workflows.

    ``LocalFS`` (below) implements this against ``pathlib.Path``.
    alpaca-ui provides a concrete remote implementation backed by asyncssh.
    """

    # ── Discovery ────────────────────────────────────────────────────────────

    def exists(self, path: Path) -> bool:
        """True if path exists (file or directory)."""
        ...

    def is_dir(self, path: Path) -> bool:
        """True if path is a directory."""
        ...

    def is_file(self, path: Path) -> bool:
        """True if path is a regular file."""
        ...

    def iterdir(self, path: Path) -> list[Path]:
        """Return immediate children of directory ``path`` (unsorted)."""
        ...

    def rglob_wavs(self, path: Path) -> list[Path]:
        """Return all .wav files under ``path``, recursively, sorted."""
        ...

    # ── Reading ──────────────────────────────────────────────────────────────

    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """Read entire file as text."""
        ...

    def open_read(self, path: Path) -> BinaryIO:
        """
        Open file for binary reading.

        The returned object MUST support ``seek()`` so that soundfile can read
        WAV headers without consuming the entire stream.
        Caller is responsible for closing (use as context manager).
        """
        ...

    def open_write(self, path: Path) -> BinaryIO:
        """Open file for binary writing, creating parent directories if needed."""
        ...

    # ── Writing ──────────────────────────────────────────────────────────────

    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        """Write text to file, creating parent directories if needed."""
        ...

    def makedirs(self, path: Path) -> None:
        """Create directory and all parents. No-op if already exists."""
        ...

    # ── Mutation (apply-rename only) ─────────────────────────────────────────

    def rename(self, src: Path, dst: Path) -> None:
        """
        Rename src to dst.

        MUST be atomic on the underlying filesystem (same-FS rename).
        MUST NOT overwrite dst if it exists — raise ``FileExistsError`` if so.
        Caller guarantees dst.parent exists before calling this.
        """
        ...

    def unlink(self, path: Path) -> None:
        """Delete file ``path``."""
        ...


# ---------------------------------------------------------------------------
# LocalFS — pathlib.Path wrapper
# ---------------------------------------------------------------------------


class LocalFS:
    """
    FileSystem implementation backed by the local filesystem via pathlib.

    This is the default used by all workflow functions when no ``fs`` is given.
    """

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def iterdir(self, path: Path) -> list[Path]:
        return list(path.iterdir())

    def rglob_wavs(self, path: Path) -> list[Path]:
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")

    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        return path.read_text(encoding=encoding)

    def open_read(self, path: Path) -> BinaryIO:
        return open(path, "rb")

    def open_write(self, path: Path) -> BinaryIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "wb")

    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    def makedirs(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def rename(self, src: Path, dst: Path) -> None:
        if dst.exists():
            raise FileExistsError(f"Target exists, refusing to overwrite: {dst}")
        src.rename(dst)

    def unlink(self, path: Path) -> None:
        path.unlink()


# Module-level default — imported by workflows.py.
# Defined here so workflows.py never needs to import LocalFS by name.
_DEFAULT_FS: FileSystem = LocalFS()

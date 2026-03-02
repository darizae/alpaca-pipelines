"""Path validation and resolution utilities."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def validate_relative_path(relative_path: str, root: Path) -> Path:
    """Resolve a root-relative path, rejecting absolute paths and traversal."""
    posix = PurePosixPath(relative_path)
    if posix.is_absolute():
        raise ValueError("Absolute path not allowed: {}".format(relative_path))
    if ".." in posix.parts:
        raise ValueError("Path traversal not allowed: {}".format(relative_path))

    resolved = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            "Path escapes root: {} resolves to {}, root is {}".format(
                relative_path, resolved, root_resolved
            )
        ) from None
    return resolved


def validate_basename(filename: str) -> str:
    """Ensure a filename is a safe basename with no separators or traversal."""
    if "/" in filename or "\\" in filename:
        raise ValueError("Filename contains path separator: {!r}".format(filename))
    if ".." in filename:
        raise ValueError("Filename contains traversal: {!r}".format(filename))
    if filename != Path(filename).name:
        raise ValueError("Filename is not a plain basename: {!r}".format(filename))
    return filename


def ensure_directory(path: Path) -> Path:
    """Create a directory and all parents, returning the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

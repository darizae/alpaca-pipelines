from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from alpaca_pipelines.datasets.contracts import Manifest, ManifestMeta, SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.io_utils import read_json, write_json
from alpaca_pipelines.datasets.paths import MANIFEST_FILENAME


def _compute_entries_hash(snippets: list[SnippetEntry]) -> str:
    serialized = json.dumps(
        [s.model_dump() for s in snippets],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_manifest(
    strategy_name: str,
    collection_root: Path,
    merged_index_path: Path,
    seed: int,
    snippets: list[SnippetEntry],
    strategy_config: dict[str, object] | None = None,
) -> Manifest:
    n_target = sum(1 for s in snippets if s.classification == "target")
    n_noise = sum(1 for s in snippets if s.classification == "noise")

    entries_hash = _compute_entries_hash(snippets)

    meta = ManifestMeta(
        strategy_name=strategy_name,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        collection_root=str(collection_root),
        merged_index_path=str(merged_index_path),
        seed=seed,
        n_snippets=len(snippets),
        n_target=n_target,
        n_noise=n_noise,
        manifest_hash=entries_hash,
        strategy_config=strategy_config,
    )

    return Manifest(meta=meta, snippets=snippets)


def write_manifest(
    manifest: Manifest,
    dataset_dir: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> Path:
    manifest_path = dataset_dir / MANIFEST_FILENAME
    payload = manifest.model_dump()
    write_json(manifest_path, payload, fs)
    return manifest_path


def load_manifest(dataset_dir: Path, fs: FileSystem = _DEFAULT_FS) -> Manifest:
    manifest_path = dataset_dir / MANIFEST_FILENAME
    data = read_json(manifest_path, fs)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {manifest_path}")
    return Manifest.model_validate(data)

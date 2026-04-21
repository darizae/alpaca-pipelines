from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from alpaca_pipelines.datasets.contracts import Manifest, ManifestMeta, SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.io_utils import read_json, write_json
from alpaca_pipelines.datasets.paths import MANIFEST_FILENAME
from alpaca_pipelines.recordings import SourceRecording, compute_recording_counts


def _compute_entries_hash(
    snippets: list[SnippetEntry],
    recordings: list[SourceRecording],
) -> str:
    serialized = json.dumps(
        {
            "snippets": [s.model_dump() for s in snippets],
            "recordings": [recording.model_dump() for recording in recordings],
        },
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
    recordings: list[SourceRecording],
    strategy_config: dict[str, object] | None = None,
) -> Manifest:
    n_target = sum(1 for s in snippets if s.classification == "target")
    n_noise = sum(1 for s in snippets if s.classification == "noise")

    entries_hash = _compute_entries_hash(snippets, recordings)
    n_recordings, n_recordings_with_sidecar = compute_recording_counts(recordings)
    provenance_summary, manual_curation_summary = _build_provenance_summaries(snippets)

    meta = ManifestMeta(
        strategy_name=strategy_name,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        collection_root=str(collection_root),
        merged_index_path=str(merged_index_path),
        seed=seed,
        n_snippets=len(snippets),
        n_target=n_target,
        n_noise=n_noise,
        n_recordings=n_recordings,
        n_recordings_with_sidecar=n_recordings_with_sidecar,
        manifest_hash=entries_hash,
        strategy_config=strategy_config,
        provenance_summary=provenance_summary,
        manual_curation_summary=manual_curation_summary,
    )

    return Manifest(meta=meta, snippets=snippets, recordings=recordings)


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


def _build_provenance_summaries(
    snippets: list[SnippetEntry],
) -> tuple[dict[str, dict[str, int] | int], dict[str, dict[str, int] | int]]:
    by_provenance_type: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_collection: dict[str, int] = {}
    by_source_recording_key: dict[str, int] = {}

    manual_by_label: dict[str, int] = {}
    manual_by_collection: dict[str, int] = {}
    manual_by_source_recording_key: dict[str, int] = {}
    manual_total = 0

    for snippet in snippets:
        provenance_type = _resolve_provenance_type(snippet)
        if provenance_type == "manual_review_curated":
            _validate_manual_review_curated_snippet(snippet)
        by_provenance_type[provenance_type] = by_provenance_type.get(provenance_type, 0) + 1
        by_label[snippet.classification] = by_label.get(snippet.classification, 0) + 1
        source_collection = snippet.source_collection_name or snippet.collection
        by_collection[source_collection] = by_collection.get(source_collection, 0) + 1
        if snippet.source_recording_key:
            by_source_recording_key[snippet.source_recording_key] = (
                by_source_recording_key.get(snippet.source_recording_key, 0) + 1
            )

        if provenance_type == "manual_review_curated":
            manual_total += 1
            manual_label = snippet.curated_label or snippet.classification
            manual_by_label[manual_label] = manual_by_label.get(manual_label, 0) + 1
            manual_by_collection[source_collection] = (
                manual_by_collection.get(source_collection, 0) + 1
            )
            if snippet.source_recording_key:
                manual_by_source_recording_key[snippet.source_recording_key] = (
                    manual_by_source_recording_key.get(snippet.source_recording_key, 0) + 1
                )

    provenance_summary: dict[str, dict[str, int] | int] = {
        "by_provenance_type": dict(sorted(by_provenance_type.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_collection": dict(sorted(by_collection.items())),
        "by_source_recording_key": dict(sorted(by_source_recording_key.items())),
        "total_manual_review_curated": manual_total,
    }
    manual_curation_summary: dict[str, dict[str, int] | int] = {
        "total_examples": manual_total,
        "by_label": dict(sorted(manual_by_label.items())),
        "by_collection": dict(sorted(manual_by_collection.items())),
        "by_source_recording_key": dict(sorted(manual_by_source_recording_key.items())),
    }
    return provenance_summary, manual_curation_summary


def _resolve_provenance_type(snippet: SnippetEntry) -> str:
    if snippet.provenance_type is not None:
        return snippet.provenance_type
    if snippet.source_type in {"hum", "low_quality_hum"}:
        return "indexed_hum"
    if snippet.source_type == "mined_source":
        return "raw_negative_source"
    if snippet.source_type == "manual_review_curated":
        return "manual_review_curated"
    return "indexed_clip"


def _validate_manual_review_curated_snippet(snippet: SnippetEntry) -> None:
    if not snippet.source_curated_example_id:
        raise ValueError(
            "manual_review_curated snippet missing source_curated_example_id: uid {}".format(
                snippet.uid
            )
        )
    if not snippet.source_review_session_id:
        raise ValueError(
            "manual_review_curated snippet missing source_review_session_id: uid {}".format(
                snippet.uid
            )
        )
    if not snippet.source_review_item_id:
        raise ValueError(
            "manual_review_curated snippet missing source_review_item_id: uid {}".format(
                snippet.uid
            )
        )

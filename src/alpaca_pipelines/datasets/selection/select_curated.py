from __future__ import annotations

import random
from collections import defaultdict
from itertools import count
from pathlib import Path

from alpaca_pipelines.datasets.contracts import SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.selection.select_positives import _copy_wav
from alpaca_pipelines.prediction.review.curated import (
    CURATED_MANIFEST_FILENAME,
    CuratedPredictionSourceItem,
    CuratedPredictionSourceManifest,
)


def select_curated_examples(
    *,
    curated_root: Path,
    snippets_dir: Path,
    uid_counter: count[int],
    filters: dict[str, list[str]] | None,
    max_examples: int | None,
    seed: int,
    fs: FileSystem = _DEFAULT_FS,
) -> tuple[list[SnippetEntry], dict[str, int]]:
    manifests = _load_curated_manifests(curated_root, fs)
    filtered_items = _apply_filters(manifests, filters)
    deduped_items, deduped_count = _dedupe_items(filtered_items)
    selected_items = _limit_items(deduped_items, max_examples=max_examples, seed=seed)

    snippets: list[SnippetEntry] = []
    for item in selected_items:
        uid = next(uid_counter)
        filename = f"{item.label}-curated_{uid:06d}_{item.source_collection_name}.wav"
        destination = snippets_dir / filename
        _copy_wav(Path(item.snippet_wav_path), destination, fs)
        snippet = SnippetEntry(
            uid=uid,
            filename=filename,
            classification=item.label,
            source_type="manual_review_curated",
            source_path=item.source_relative_path,
            start_s=0.0,
            end_s=item.duration_s,
            duration_s=item.duration_s,
            quality=None,
            subject_id=None,
            recording_date=None,
            collection=item.source_collection_name,
            session_key=None,
            source_recording_key=item.source_recording_key,
            provenance_type="manual_review_curated",
            curated_label=item.label,
            source_collection_name=item.source_collection_name,
            source_category_dir=item.source_category_dir,
            source_relative_path=item.source_relative_path,
            source_prediction_run_id=item.prediction_run_id,
            source_review_session_id=item.review_session_id,
            source_review_item_id=item.review_item_id,
            source_curated_example_id=item.curated_example_id,
        )
        snippets.append(snippet)

    summary = {
        "curated_candidates": len(filtered_items),
        "curated_selected": len(selected_items),
        "curated_deduped": deduped_count,
    }
    return snippets, summary


def _load_curated_manifests(
    curated_root: Path, fs: FileSystem
) -> list[CuratedPredictionSourceManifest]:
    if not fs.exists(curated_root):
        return []

    manifests: list[CuratedPredictionSourceManifest] = []
    for manifest_path in _find_manifest_paths(curated_root, fs):
        payload = fs.read_text(manifest_path)
        manifest = CuratedPredictionSourceManifest.model_validate_json(payload)
        manifests.append(manifest)
    return manifests


def _find_manifest_paths(root: Path, fs: FileSystem) -> list[Path]:
    if not fs.is_dir(root):
        return []
    discovered: list[Path] = []
    for entry in sorted(fs.iterdir(root)):
        if fs.is_dir(entry):
            discovered.extend(_find_manifest_paths(entry, fs))
        elif entry.name == CURATED_MANIFEST_FILENAME:
            discovered.append(entry)
    return discovered


def _apply_filters(
    manifests: list[CuratedPredictionSourceManifest],
    filters: dict[str, list[str]] | None,
) -> list[CuratedPredictionSourceItem]:
    collection_names = set(filters.get("collection_names", [])) if filters else set()
    labels = set(filters.get("labels", [])) if filters else set()
    prediction_run_ids = set(filters.get("prediction_run_ids", [])) if filters else set()
    source_recording_keys = set(filters.get("source_recording_keys", [])) if filters else set()

    selected: list[CuratedPredictionSourceItem] = []
    for manifest in manifests:
        for item in manifest.items:
            if collection_names and item.source_collection_name not in collection_names:
                continue
            if labels and item.label not in labels:
                continue
            if prediction_run_ids and item.prediction_run_id not in prediction_run_ids:
                continue
            if source_recording_keys and item.source_recording_key not in source_recording_keys:
                continue
            selected.append(item)
    return selected


def _dedupe_items(
    items: list[CuratedPredictionSourceItem],
) -> tuple[list[CuratedPredictionSourceItem], int]:
    grouped: dict[tuple[str, float, float, str], list[CuratedPredictionSourceItem]] = defaultdict(
        list
    )
    for item in items:
        dedupe_key = (
            item.source_recording_key,
            round(item.start_s, 6),
            round(item.end_s, 6),
            item.label,
        )
        grouped[dedupe_key].append(item)

    deduped: list[CuratedPredictionSourceItem] = []
    duplicates = 0
    for candidates in grouped.values():
        deduped.append(sorted(candidates, key=lambda candidate: candidate.curated_example_id)[0])
        duplicates += max(len(candidates) - 1, 0)
    deduped.sort(key=lambda item: item.curated_example_id)
    return deduped, duplicates


def _limit_items(
    items: list[CuratedPredictionSourceItem],
    *,
    max_examples: int | None,
    seed: int,
) -> list[CuratedPredictionSourceItem]:
    if max_examples is None or len(items) <= max_examples:
        return items
    rng = random.Random(seed)
    selected = items.copy()
    rng.shuffle(selected)
    limited = selected[:max_examples]
    limited.sort(key=lambda item: item.curated_example_id)
    return limited

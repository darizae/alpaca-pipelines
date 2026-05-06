from __future__ import annotations

import random
from collections import defaultdict
from itertools import count
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from alpaca_pipelines.datasets.contracts import SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.selection.select_positives import _copy_wav
from alpaca_pipelines.prediction.review.curated import (
    CURATED_CATEGORY_NAMES,
    CURATED_MANIFEST_FILENAME,
    CuratedPredictionSourceItem,
    CuratedPredictionSourceManifest,
)

_CURATED_DEDUPE_KEY_POLICY = "source_recording_key|round(start_s,6)|round(end_s,6)|label"


class _CuratedManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    item: CuratedPredictionSourceItem


def select_curated_examples(
    *,
    collection_root: Path,
    snippets_dir: Path,
    uid_counter: count[int],
    filters: dict[str, list[str]] | None,
    max_examples: int | None,
    seed: int,
    fs: FileSystem = _DEFAULT_FS,
) -> tuple[list[SnippetEntry], dict[str, int | str | bool]]:
    manifest_items = _load_curated_manifest_items(collection_root, fs)
    filtered_items = _apply_filters(manifest_items, filters)
    dedupe_result = _dedupe_items(filtered_items)
    selected_items = _limit_items(
        dedupe_result.deduped,
        max_examples=max_examples,
        seed=seed,
    )

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
            source_path=item.source_display_path,
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
            source_display_path=item.source_display_path,
            source_prediction_run_id=item.prediction_run_id,
            source_review_session_id=item.review_session_id,
            source_review_item_id=item.review_item_id,
            source_curated_example_id=item.curated_example_id,
        )
        snippets.append(snippet)

    summary: dict[str, int | str | bool] = {
        "curated_candidates": len(filtered_items),
        "curated_selected": len(selected_items),
        "curated_deduped": dedupe_result.duplicates_removed,
        "curated_duplicates_removed": dedupe_result.duplicates_removed,
        "curated_dedupe_key_policy": _CURATED_DEDUPE_KEY_POLICY,
        "curated_duplicates_crossed_source_manifests": dedupe_result.cross_manifest_duplicates,
    }
    return snippets, summary


class _DedupeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deduped: list[CuratedPredictionSourceItem]
    duplicates_removed: int
    cross_manifest_duplicates: bool


def _load_curated_manifest_items(
    collection_root: Path, fs: FileSystem
) -> list[_CuratedManifestItem]:
    if not fs.exists(collection_root):
        return []

    manifest_items: list[_CuratedManifestItem] = []
    for manifest_path in _find_manifest_paths(collection_root, fs):
        payload = fs.read_text(manifest_path)
        manifest = CuratedPredictionSourceManifest.model_validate_json(payload)
        for item in manifest.items:
            manifest_items.append(_CuratedManifestItem(manifest_path=str(manifest_path), item=item))
    return manifest_items


def _find_manifest_paths(root: Path, fs: FileSystem) -> list[Path]:
    if not fs.is_dir(root):
        return []
    discovered: list[Path] = []
    for collection_dir in sorted(fs.iterdir(root)):
        if not fs.is_dir(collection_dir):
            continue
        for category_name in CURATED_CATEGORY_NAMES:
            category_dir = collection_dir / category_name
            if not fs.is_dir(category_dir):
                continue
            discovered.extend(_find_manifest_paths_under(category_dir, fs))
    return discovered


def _find_manifest_paths_under(root: Path, fs: FileSystem) -> list[Path]:
    if not fs.is_dir(root):
        return []
    discovered: list[Path] = []
    for entry in sorted(fs.iterdir(root)):
        if fs.is_dir(entry):
            discovered.extend(_find_manifest_paths_under(entry, fs))
        elif entry.name == CURATED_MANIFEST_FILENAME:
            discovered.append(entry)
    return discovered


def _apply_filters(
    manifest_items: list[_CuratedManifestItem],
    filters: dict[str, list[str]] | None,
) -> list[_CuratedManifestItem]:
    collection_names = set(filters.get("collection_names", [])) if filters else set()
    labels = set(filters.get("labels", [])) if filters else set()
    prediction_run_ids = set(filters.get("prediction_run_ids", [])) if filters else set()
    source_recording_keys = set(filters.get("source_recording_keys", [])) if filters else set()

    selected: list[_CuratedManifestItem] = []
    for manifest_item in manifest_items:
        item = manifest_item.item
        if collection_names and item.source_collection_name not in collection_names:
            continue
        if labels and item.label not in labels:
            continue
        if prediction_run_ids and item.prediction_run_id not in prediction_run_ids:
            continue
        if source_recording_keys and item.source_recording_key not in source_recording_keys:
            continue
        selected.append(manifest_item)
    return selected


def _dedupe_items(items: list[_CuratedManifestItem]) -> _DedupeResult:
    grouped: dict[tuple[str, float, float, str], list[_CuratedManifestItem]] = defaultdict(list)
    for manifest_item in items:
        item = manifest_item.item
        dedupe_key = (
            item.source_recording_key,
            round(item.start_s, 6),
            round(item.end_s, 6),
            item.label,
        )
        grouped[dedupe_key].append(manifest_item)

    deduped: list[CuratedPredictionSourceItem] = []
    duplicates_removed = 0
    cross_manifest_duplicates = False
    for candidates in grouped.values():
        deduped.append(
            sorted(candidates, key=lambda candidate: candidate.item.curated_example_id)[0].item
        )
        duplicates_removed += max(len(candidates) - 1, 0)
        if len({candidate.manifest_path for candidate in candidates}) > 1:
            cross_manifest_duplicates = True

    deduped.sort(key=lambda item: item.curated_example_id)
    return _DedupeResult(
        deduped=deduped,
        duplicates_removed=duplicates_removed,
        cross_manifest_duplicates=cross_manifest_duplicates,
    )


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

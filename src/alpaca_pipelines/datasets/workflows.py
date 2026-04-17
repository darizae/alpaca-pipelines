from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from alpaca_pipelines.datasets.config import DatasetBuildConfig, StrategyConfig
from alpaca_pipelines.datasets.contracts import Manifest, SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.index_reader import IndexPayload, load_merged_index
from alpaca_pipelines.datasets.io_utils import read_json, write_csv_rows
from alpaca_pipelines.datasets.manifest import (
    _build_provenance_summaries,
    _compute_entries_hash,
    build_manifest,
    load_manifest,
    write_manifest,
)
from alpaca_pipelines.datasets.mining.mine_negatives import mine_negatives_for_positives
from alpaca_pipelines.datasets.paths import (
    SNIPPETS_DIR,
    SPLITS_DIR,
    ensure_dataset_dirs,
)
from alpaca_pipelines.datasets.review.apply_review import apply_review_table
from alpaca_pipelines.datasets.review.concatenate import prepare_review_artifacts
from alpaca_pipelines.datasets.selection.select_curated import select_curated_examples
from alpaca_pipelines.datasets.selection.select_positives import (
    select_low_quality_as_negatives,
    select_positives,
)
from alpaca_pipelines.datasets.source_discovery import discover_source_files
from alpaca_pipelines.datasets.splitting.strategies import apply_split
from alpaca_pipelines.prediction.review.curated import curated_sources_root


@dataclass(frozen=True)
class BuildResult:
    dataset_dir: Path
    manifest: Manifest
    n_target: int
    n_noise: int
    splits: dict[str, int]
    curated_summary: dict[str, int]


@dataclass(frozen=True)
class ReviewPrepResult:
    dataset_dir: Path
    target_selection_table_path: Path
    noise_selection_table_path: Path
    target_concat_wav_path: Path
    noise_concat_wav_path: Path
    n_target_snippets: int
    n_noise_snippets: int


@dataclass(frozen=True)
class ReviewApplyResult:
    dataset_dir: Path
    n_corrections: int
    n_discarded: int
    updated_manifest: Manifest


def load_build_config(config_path: Path, fs: FileSystem = _DEFAULT_FS) -> DatasetBuildConfig:
    data = read_json(config_path, fs)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {config_path}")
    return DatasetBuildConfig.model_validate(data)


def build_dataset(
    strategy_name: str,
    strategy_config: StrategyConfig,
    collection_root: Path,
    merged_index_path: Path,
    datasets_root: Path,
    index_payload: IndexPayload | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> BuildResult:
    if index_payload is None:
        index_payload = load_merged_index(merged_index_path, fs)
    recordings_by_key = {recording.key: recording for recording in index_payload.recordings}

    sorted_entries = sorted(
        index_payload.entries,
        key=lambda e: (e.collection, e.subject_id, e.recording_date, e.hum_uid),
    )
    target_collection_names = set(strategy_config.target_collection_names)
    noise_collection_names = set(strategy_config.noise_collection_names)
    target_entries = [
        entry for entry in sorted_entries if entry.collection in target_collection_names
    ]
    noise_entries = [
        entry for entry in sorted_entries if entry.collection in noise_collection_names
    ]

    dataset_dir = datasets_root / strategy_name
    if fs.exists(dataset_dir):
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_dir}. Remove it first to rebuild."
        )

    ensure_dataset_dirs(dataset_dir, fs)
    snippets_dir = dataset_dir / SNIPPETS_DIR

    uid_counter: count[int] = count(1)

    positive_snippets = select_positives(
        entries=target_entries,
        recordings_by_key=recordings_by_key,
        min_quality=strategy_config.min_quality,
        collection_root=collection_root,
        snippets_dir=snippets_dir,
        uid_counter=uid_counter,
        duration_tolerance_s=strategy_config.duration_tolerance_s,
        fs=fs,
    )
    curated_summary: dict[str, int] = {
        "curated_candidates": 0,
        "curated_selected": 0,
        "curated_deduped": 0,
    }
    curated_snippets: list[SnippetEntry] = []
    if strategy_config.include_manual_review_curated:
        curated_snippets, curated_summary = select_curated_examples(
            curated_root=curated_sources_root(datasets_root=datasets_root),
            snippets_dir=snippets_dir,
            uid_counter=uid_counter,
            filters=strategy_config.manual_review_curated_filters.model_dump(),
            max_examples=strategy_config.manual_review_curated_max_examples,
            seed=strategy_config.seed,
            fs=fs,
        )

    all_positive_snippets = [
        *positive_snippets,
        *[snippet for snippet in curated_snippets if snippet.classification == "target"],
    ]
    if not all_positive_snippets:
        raise ValueError(
            "Dataset build requires labelled target examples, but the target pool is empty."
        )

    low_quality_negatives: list[SnippetEntry] = []
    if strategy_config.noise_mining.low_quality_as_negative:
        low_quality_negatives = select_low_quality_as_negatives(
            entries=noise_entries,
            recordings_by_key=recordings_by_key,
            low_quality_threshold=strategy_config.noise_mining.low_quality_threshold,
            collection_root=collection_root,
            snippets_dir=snippets_dir,
            uid_counter=uid_counter,
            duration_tolerance_s=strategy_config.duration_tolerance_s,
            fs=fs,
        )

    source_files = discover_source_files(
        collection_root=collection_root,
        collection_names=strategy_config.noise_collection_names,
        source_category_dirs=strategy_config.noise_mining.source_category_dirs,
        fs=fs,
    )

    mined_negatives = mine_negatives_for_positives(
        positive_snippets=all_positive_snippets,
        source_files=source_files,
        noise_per_positive=strategy_config.noise_per_positive,
        attempts_per_slot=strategy_config.noise_mining.attempts_per_slot,
        snippets_dir=snippets_dir,
        uid_counter=uid_counter,
        seed=strategy_config.seed,
        fs=fs,
    )

    curated_noise_snippets = [
        snippet for snippet in curated_snippets if snippet.classification == "noise"
    ]

    all_snippets = [
        *all_positive_snippets,
        *curated_noise_snippets,
        *low_quality_negatives,
        *mined_negatives,
    ]

    split_result = apply_split(
        snippets=all_snippets,
        strategy_name=strategy_config.split_strategy,
        seed=strategy_config.seed,
        fractions=strategy_config.split_fractions,
    )

    annotated_snippets: list[SnippetEntry] = []
    for split_name, split_snippets in split_result.items():
        for snippet in split_snippets:
            annotated_snippets.append(snippet.model_copy(update={"split": split_name}))

    manifest_recordings = _select_manifest_recordings(
        annotated_snippets,
        recordings_by_key,
    )

    _write_split_csvs(dataset_dir, split_result, fs)

    manifest = build_manifest(
        strategy_name=strategy_name,
        collection_root=collection_root,
        merged_index_path=merged_index_path,
        seed=strategy_config.seed,
        snippets=annotated_snippets,
        recordings=manifest_recordings,
        strategy_config=strategy_config.model_dump(),
    )
    write_manifest(manifest, dataset_dir, fs)

    split_counts = {name: len(snippets) for name, snippets in split_result.items()}

    return BuildResult(
        dataset_dir=dataset_dir,
        manifest=manifest,
        n_target=manifest.meta.n_target,
        n_noise=manifest.meta.n_noise,
        splits=split_counts,
        curated_summary=curated_summary,
    )


def prepare_review(dataset_dir: Path, fs: FileSystem = _DEFAULT_FS) -> ReviewPrepResult:
    manifest = load_manifest(dataset_dir, fs)

    review_config = _read_review_config_from_manifest(manifest)
    gap_seconds = float(review_config.get("review_gap_s", 0.5))
    freq_low_hz = int(review_config.get("freq_low_hz", 0))
    freq_high_hz = int(review_config.get("freq_high_hz", 4000))

    artifacts = prepare_review_artifacts(
        dataset_dir=dataset_dir,
        manifest=manifest,
        gap_seconds=gap_seconds,
        freq_low_hz=freq_low_hz,
        freq_high_hz=freq_high_hz,
        fs=fs,
    )

    return ReviewPrepResult(
        dataset_dir=dataset_dir,
        target_selection_table_path=artifacts["target"].selection_table_path,
        noise_selection_table_path=artifacts["noise"].selection_table_path,
        target_concat_wav_path=artifacts["target"].concat_wav_path,
        noise_concat_wav_path=artifacts["noise"].concat_wav_path,
        n_target_snippets=artifacts["target"].n_snippets,
        n_noise_snippets=artifacts["noise"].n_snippets,
    )


def apply_review(
    dataset_dir: Path,
    target_review_table_path: Path,
    noise_review_table_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> ReviewApplyResult:
    manifest = load_manifest(dataset_dir, fs)

    updated_manifest, n_reclassified, n_discarded = apply_review_table(
        dataset_dir,
        manifest,
        target_review_table_path,
        noise_review_table_path,
        fs,
    )

    updated_manifest = _recompute_manifest_hash(updated_manifest)
    write_manifest(updated_manifest, dataset_dir, fs)
    _regenerate_split_csvs(dataset_dir, updated_manifest, fs)

    return ReviewApplyResult(
        dataset_dir=dataset_dir,
        n_corrections=n_reclassified,
        n_discarded=n_discarded,
        updated_manifest=updated_manifest,
    )


def run_active_strategies(
    config_path: Path,
    collection_root: Path,
    merged_index_path: Path,
    datasets_root: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> list[BuildResult]:
    build_config = load_build_config(config_path, fs)
    index_payload = load_merged_index(merged_index_path, fs)

    results: list[BuildResult] = []
    for strategy_name in build_config.active_strategies:
        if strategy_name not in build_config.strategies:
            raise ValueError(f"Active strategy '{strategy_name}' not found in strategies config")
        strategy_config = build_config.strategies[strategy_name]
        result = build_dataset(
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            collection_root=collection_root,
            merged_index_path=merged_index_path,
            datasets_root=datasets_root,
            index_payload=index_payload,
            fs=fs,
        )
        results.append(result)

    return results


def _write_split_csvs(
    dataset_dir: Path,
    split_result: dict[str, list[SnippetEntry]],
    fs: FileSystem,
) -> None:
    splits_dir = dataset_dir / SPLITS_DIR
    fs.makedirs(splits_dir)

    for split_name, snippets in split_result.items():
        csv_path = splits_dir / f"{split_name}.csv"
        lines = [[s.filename] for s in snippets]
        write_csv_rows(csv_path, lines, fs)


def _regenerate_split_csvs(dataset_dir: Path, manifest: Manifest, fs: FileSystem) -> None:
    split_groups: dict[str, list[SnippetEntry]] = {"train": [], "val": [], "test": []}
    for snippet in manifest.snippets:
        if snippet.split and snippet.split in split_groups:
            split_groups[snippet.split].append(snippet)
    _write_split_csvs(dataset_dir, split_groups, fs)


def _recompute_manifest_hash(manifest: Manifest) -> Manifest:
    new_hash = _compute_entries_hash(manifest.snippets, manifest.recordings)
    provenance_summary, manual_curation_summary = _build_provenance_summaries(manifest.snippets)
    return manifest.model_copy(
        update={
            "meta": manifest.meta.model_copy(
                update={
                    "manifest_hash": new_hash,
                    "provenance_summary": provenance_summary,
                    "manual_curation_summary": manual_curation_summary,
                }
            ),
        }
    )


def _read_review_config_from_manifest(manifest: Manifest) -> dict[str, Any]:
    stored_config = manifest.meta.strategy_config
    if stored_config is not None:
        return {
            "review_gap_s": stored_config.get("review_gap_s", 0.5),
            "freq_low_hz": stored_config.get("freq_low_hz", 0),
            "freq_high_hz": stored_config.get("freq_high_hz", 4000),
        }
    return {
        "review_gap_s": 0.5,
        "freq_low_hz": 0,
        "freq_high_hz": 4000,
    }


def _select_manifest_recordings(
    snippets: list[SnippetEntry],
    recordings_by_key: dict[str, Any],
) -> list[Any]:
    selected_keys = {
        snippet.source_recording_key for snippet in snippets if snippet.source_recording_key
    }
    return [recordings_by_key[key] for key in sorted(selected_keys) if key in recordings_by_key]

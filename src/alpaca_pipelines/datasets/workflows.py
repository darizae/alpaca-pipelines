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
    _compute_entries_hash,
    build_manifest,
    load_manifest,
    write_manifest,
)
from alpaca_pipelines.datasets.mining.mine_negatives import mine_negatives_for_positives
from alpaca_pipelines.datasets.paths import (
    REVIEW_CONCAT_FILENAME,
    REVIEW_DIR,
    SNIPPETS_DIR,
    SPLITS_DIR,
    ensure_dataset_dirs,
)
from alpaca_pipelines.datasets.review.apply_review import apply_review_table
from alpaca_pipelines.datasets.review.concatenate import prepare_review_artifacts
from alpaca_pipelines.datasets.selection.select_positives import (
    select_low_quality_as_negatives,
    select_positives,
)
from alpaca_pipelines.datasets.source_discovery import discover_source_files
from alpaca_pipelines.datasets.splitting.strategies import apply_split


@dataclass(frozen=True)
class BuildResult:
    dataset_dir: Path
    manifest: Manifest
    n_target: int
    n_noise: int
    splits: dict[str, int]


@dataclass(frozen=True)
class ReviewPrepResult:
    dataset_dir: Path
    selection_table_path: Path
    concat_wav_path: Path


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

    sorted_entries = sorted(
        index_payload.entries,
        key=lambda e: (e.collection, e.subject_id, e.recording_date, e.hum_uid),
    )

    dataset_dir = datasets_root / strategy_name
    if fs.exists(dataset_dir):
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_dir}. Remove it first to rebuild."
        )

    ensure_dataset_dirs(dataset_dir, fs)
    snippets_dir = dataset_dir / SNIPPETS_DIR

    uid_counter: count[int] = count(1)

    positive_snippets = select_positives(
        entries=sorted_entries,
        min_quality=strategy_config.min_quality,
        collection_root=collection_root,
        snippets_dir=snippets_dir,
        uid_counter=uid_counter,
        duration_tolerance_s=strategy_config.duration_tolerance_s,
        fs=fs,
    )

    low_quality_negatives: list[SnippetEntry] = []
    if strategy_config.noise_mining.low_quality_as_negative:
        low_quality_negatives = select_low_quality_as_negatives(
            entries=sorted_entries,
            low_quality_threshold=strategy_config.noise_mining.low_quality_threshold,
            collection_root=collection_root,
            snippets_dir=snippets_dir,
            uid_counter=uid_counter,
            duration_tolerance_s=strategy_config.duration_tolerance_s,
            fs=fs,
        )

    source_files = discover_source_files(
        collection_root=collection_root,
        source_category_dirs=strategy_config.noise_mining.source_category_dirs,
        fs=fs,
    )

    mined_negatives = mine_negatives_for_positives(
        positive_snippets=positive_snippets,
        source_files=source_files,
        noise_per_positive=strategy_config.noise_per_positive,
        attempts_per_slot=strategy_config.noise_mining.attempts_per_slot,
        snippets_dir=snippets_dir,
        uid_counter=uid_counter,
        seed=strategy_config.seed,
        fs=fs,
    )

    all_snippets = [*positive_snippets, *low_quality_negatives, *mined_negatives]

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

    _write_split_csvs(dataset_dir, split_result, fs)

    manifest = build_manifest(
        strategy_name=strategy_name,
        collection_root=collection_root,
        merged_index_path=merged_index_path,
        seed=strategy_config.seed,
        snippets=annotated_snippets,
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
    )


def prepare_review(dataset_dir: Path, fs: FileSystem = _DEFAULT_FS) -> ReviewPrepResult:
    manifest = load_manifest(dataset_dir, fs)

    review_config = _read_review_config_from_manifest(manifest)
    gap_seconds = float(review_config.get("review_gap_s", 0.5))
    freq_low_hz = int(review_config.get("freq_low_hz", 0))
    freq_high_hz = int(review_config.get("freq_high_hz", 4000))

    table_path = prepare_review_artifacts(
        dataset_dir=dataset_dir,
        manifest=manifest,
        gap_seconds=gap_seconds,
        freq_low_hz=freq_low_hz,
        freq_high_hz=freq_high_hz,
        fs=fs,
    )

    concat_path = dataset_dir / REVIEW_DIR / REVIEW_CONCAT_FILENAME

    return ReviewPrepResult(
        dataset_dir=dataset_dir,
        selection_table_path=table_path,
        concat_wav_path=concat_path,
    )


def apply_review(
    dataset_dir: Path,
    review_table_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> ReviewApplyResult:
    manifest = load_manifest(dataset_dir, fs)

    updated_manifest, n_reclassified, n_discarded = apply_review_table(
        dataset_dir,
        manifest,
        review_table_path,
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
    new_hash = _compute_entries_hash(manifest.snippets)
    return manifest.model_copy(
        update={
            "meta": manifest.meta.model_copy(update={"manifest_hash": new_hash}),
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

from __future__ import annotations

from pathlib import Path

import pytest

from alpaca_pipelines.datasets.contracts import Manifest, ManifestMeta, SnippetEntry
from alpaca_pipelines.datasets.review.apply_review import apply_review_table


def _build_manifest() -> Manifest:
    return Manifest(
        meta=ManifestMeta(
            strategy_name="dataset-a",
            created_at="2026-04-11T00:00:00Z",
            collection_root="/collections",
            merged_index_path="/collections/merged_index.json",
            seed=42,
            n_snippets=2,
            n_target=1,
            n_noise=1,
            n_recordings=0,
            n_recordings_with_sidecar=0,
            manifest_hash="abc123",
            strategy_config=None,
        ),
        snippets=[
            SnippetEntry(
                uid=1,
                filename="target-Q2_000001_audio_collection_alpha.wav",
                classification="target",
                source_type="hum",
                source_path="audio_collection_alpha/clips_labelled/a.wav",
                start_s=0.0,
                end_s=0.5,
                duration_s=0.5,
                quality=2,
                subject_id="subject-a",
                recording_date="2026-04-11",
                collection="audio_collection_alpha",
                session_key=None,
            ),
            SnippetEntry(
                uid=2,
                filename="noise-bg_000002_audio_collection_alpha.wav",
                classification="noise",
                source_type="mined_source",
                source_path="audio_collection_alpha/raw_recordings/b.wav",
                start_s=1.0,
                end_s=1.5,
                duration_s=0.5,
                quality=2,
                subject_id="subject-b",
                recording_date="2026-04-11",
                collection="audio_collection_alpha",
                session_key=None,
            ),
        ],
    )


def _seed_dataset_files(dataset_dir: Path, manifest: Manifest) -> None:
    snippets_dir = dataset_dir / "snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    for snippet in manifest.snippets:
        (snippets_dir / snippet.filename).write_bytes(b"RIFF")


def _write_table(path: Path, rows: list[tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["uid\tSound_type\treview_label"]
    for uid, sound_type, review_label in rows:
        lines.append(f"{uid}\t{sound_type}\t{review_label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raven_lite_table(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
        "High Freq (Hz)\tDelta Time (s)\tDelta Freq (Hz)\tAvg Power Density (dB FS/Hz)\t"
        "Sound_type"
    ]
    for selection, sound_type in rows:
        lines.append(f"{selection}\tSpectrogram 1\t1\t0\t1\t0\t4000\t1\t4000\t-70\t{sound_type}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_apply_review_table_accepts_identical_duplicate_rows_from_raven_views(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset-a"
    manifest = _build_manifest()
    _seed_dataset_files(dataset_dir, manifest)

    target_table = dataset_dir / "review" / "review_target_selection_table.txt"
    noise_table = dataset_dir / "review" / "review_noise_selection_table.txt"
    _write_table(
        target_table,
        [
            (1, "target", "target"),
            (1, "target", "target"),
        ],
    )
    _write_table(noise_table, [(2, "noise", "noise")])

    updated_manifest, n_reclassified, n_discarded = apply_review_table(
        dataset_dir,
        manifest,
        target_table,
        noise_table,
    )

    assert updated_manifest.meta.n_snippets == 2
    assert n_reclassified == 0
    assert n_discarded == 0


def test_apply_review_table_rejects_conflicting_duplicate_rows_for_same_uid(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset-a"
    manifest = _build_manifest()
    _seed_dataset_files(dataset_dir, manifest)

    target_table = dataset_dir / "review" / "review_target_selection_table.txt"
    noise_table = dataset_dir / "review" / "review_noise_selection_table.txt"
    _write_table(
        target_table,
        [
            (1, "target", "target"),
            (1, "target", "discard"),
        ],
    )
    _write_table(noise_table, [(2, "noise", "noise")])

    with pytest.raises(ValueError, match="conflicting labels for uid 1"):
        apply_review_table(
            dataset_dir,
            manifest,
            target_table,
            noise_table,
        )


def test_apply_review_table_accepts_raven_lite_format_and_numeric_labels(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset-a"
    manifest = _build_manifest()
    _seed_dataset_files(dataset_dir, manifest)

    target_table = dataset_dir / "review" / "review_target_selection_table.txt"
    noise_table = dataset_dir / "review" / "review_noise_selection_table.txt"
    _write_raven_lite_table(
        target_table,
        [
            (1, "0"),
            (1, "0"),
        ],
    )
    _write_raven_lite_table(
        noise_table,
        [
            (1, "1"),
            (1, "1"),
        ],
    )

    updated_manifest, n_reclassified, n_discarded = apply_review_table(
        dataset_dir,
        manifest,
        target_table,
        noise_table,
    )

    assert updated_manifest.meta.n_target == 1
    assert updated_manifest.meta.n_noise == 1
    assert n_reclassified == 2
    assert n_discarded == 0

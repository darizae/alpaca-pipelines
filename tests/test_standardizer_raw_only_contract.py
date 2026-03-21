from __future__ import annotations

from pathlib import Path

import pytest
from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import LocalFS
from alpaca_pipelines.collections.paths import CategoryNames
from alpaca_pipelines.collections.planning.rename_plan import plan_renames_for_collection
from alpaca_pipelines.collections.workflows import (
    apply_rename_plan_payload,
    build_indexes,
    scan_root,
)
from alpaca_pipelines.datasets.config import StrategyConfig
from alpaca_pipelines.datasets.workflows import build_dataset
from alpaca_pipelines.io_utils import write_json
from alpaca_pipelines.recordings import (
    SourceRecording,
    load_collection_recordings,
    write_collection_recordings,
)


def _identity_map() -> IdentityMap:
    return IdentityMap.model_validate(
        {
            "canonical": {
                "401": {"display_name": "401"},
                "402": {"display_name": "402"},
            },
            "aliases": {},
        }
    )


def _source_recording(
    *,
    key: str,
    collection: str,
    wav_path: str,
    csv_path: str | None = None,
) -> SourceRecording:
    return SourceRecording(
        key=key,
        collection=collection,
        subject_id=key.split("_", 1)[0],
        wav_path=wav_path,
        csv_path=csv_path,
    )


def test_scan_root_reports_explicit_status_matrix(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    root.mkdir()

    ready = root / "audio_collection_ready"
    (ready / "clips_labelled").mkdir(parents=True)
    (ready / "hums_segmented").mkdir()

    raw_only = root / "audio_collection_raw_only"
    (raw_only / "raw_recordings").mkdir(parents=True)

    clips_only = root / "audio_collection_clips_only"
    (clips_only / "clips_labelled").mkdir(parents=True)

    hums_only = root / "audio_collection_hums_only"
    (hums_only / "hums_segmented").mkdir(parents=True)

    empty = root / "audio_collection_empty"
    empty.mkdir()

    report = scan_root(root)
    status_by_collection = {
        str(item["collection"]): str(item["status"]) for item in report.payload["collections"]
    }

    assert status_by_collection == {
        "audio_collection_clips_only": "clips_only",
        "audio_collection_empty": "empty",
        "audio_collection_hums_only": "hums_only",
        "audio_collection_raw_only": "raw_only",
        "audio_collection_ready": "ready",
    }


def test_plan_renames_includes_raw_recording_canonicalization(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    collection_dir = root / "audio_collection_alpha"
    raw_dir = collection_dir / "raw_recordings"
    raw_dir.mkdir(parents=True)

    (raw_dir / "20250211_075558.WAV").write_bytes(b"WAV")
    (raw_dir / "20250211_075558.CSV").write_text("csv", encoding="utf-8")
    write_collection_recordings(
        collection_dir,
        [
            _source_recording(
                key="401_20250211_075558",
                collection=collection_dir.name,
                wav_path=("audio_collection_alpha/raw_recordings/20250211_075558.WAV"),
                csv_path=("audio_collection_alpha/raw_recordings/20250211_075558.CSV"),
            )
        ],
    )

    ops, _, _, raw_audit, updates = plan_renames_for_collection(
        collection_dir=collection_dir,
        identity_map=_identity_map(),
        category_names=CategoryNames(),
        fs=LocalFS(),
    )

    destinations = {Path(op.dst).name for op in ops}
    assert destinations == {"401_20250211_075558.WAV", "401_20250211_075558.CSV"}
    assert {row.kind for row in raw_audit} == {"wav", "csv"}
    assert [update.__dict__ for update in updates] == [
        {
            "collection": "audio_collection_alpha",
            "recording_key": "401_20250211_075558",
            "wav_path": "audio_collection_alpha/raw_recordings/401_20250211_075558.WAV",
            "csv_path": "audio_collection_alpha/raw_recordings/401_20250211_075558.CSV",
        }
    ]


def test_plan_renames_fails_when_raw_file_is_untracked(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    collection_dir = root / "audio_collection_alpha"
    raw_dir = collection_dir / "raw_recordings"
    raw_dir.mkdir(parents=True)

    (raw_dir / "20250211_075558.WAV").write_bytes(b"WAV")
    (raw_dir / "EXTRA.WAV").write_bytes(b"WAV")
    write_collection_recordings(
        collection_dir,
        [
            _source_recording(
                key="401_20250211_075558",
                collection=collection_dir.name,
                wav_path=("audio_collection_alpha/raw_recordings/20250211_075558.WAV"),
            )
        ],
    )

    with pytest.raises(ValueError, match="not represented in recordings.json"):
        plan_renames_for_collection(
            collection_dir=collection_dir,
            identity_map=_identity_map(),
            category_names=CategoryNames(),
            fs=LocalFS(),
        )


def test_apply_rename_plan_rolls_back_file_and_recordings_updates(tmp_path: Path) -> None:
    class FailOnSecondRecordingsWriteFS(LocalFS):
        def __init__(self) -> None:
            self.recordings_write_calls = 0

        def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
            if path.name == "recordings.json":
                self.recordings_write_calls += 1
                if self.recordings_write_calls == 2:
                    raise RuntimeError("synthetic recordings write failure")
            super().write_text(path, content, encoding=encoding)

    root = tmp_path / "collection"
    root.mkdir()
    fs = FailOnSecondRecordingsWriteFS()

    collection_a = root / "audio_collection_a"
    collection_b = root / "audio_collection_b"
    raw_a = collection_a / "raw_recordings"
    raw_b = collection_b / "raw_recordings"
    raw_a.mkdir(parents=True)
    raw_b.mkdir(parents=True)

    old_a = raw_a / "20250211_075558.WAV"
    old_b = raw_b / "20250212_081000.WAV"
    old_a.write_bytes(b"A")
    old_b.write_bytes(b"B")

    write_collection_recordings(
        collection_a,
        [
            _source_recording(
                key="401_20250211_075558",
                collection=collection_a.name,
                wav_path="audio_collection_a/raw_recordings/20250211_075558.WAV",
            )
        ],
    )
    write_collection_recordings(
        collection_b,
        [
            _source_recording(
                key="402_20250212_081000",
                collection=collection_b.name,
                wav_path="audio_collection_b/raw_recordings/20250212_081000.WAV",
            )
        ],
    )

    payload = {
        "root": str(root),
        "ops": [
            {
                "src": str(old_a),
                "dst": str(raw_a / "401_20250211_075558.WAV"),
            },
            {
                "src": str(old_b),
                "dst": str(raw_b / "402_20250212_081000.WAV"),
            },
        ],
        "recordings_updates": [
            {
                "collection": "audio_collection_a",
                "recording_key": "401_20250211_075558",
                "wav_path": "audio_collection_a/raw_recordings/401_20250211_075558.WAV",
                "csv_path": None,
            },
            {
                "collection": "audio_collection_b",
                "recording_key": "402_20250212_081000",
                "wav_path": "audio_collection_b/raw_recordings/402_20250212_081000.WAV",
                "csv_path": None,
            },
        ],
    }

    with pytest.raises(RuntimeError, match="Filesystem changes were rolled back"):
        apply_rename_plan_payload(payload=payload, fs=fs)

    assert old_a.is_file()
    assert old_b.is_file()
    assert not (raw_a / "401_20250211_075558.WAV").exists()
    assert not (raw_b / "402_20250212_081000.WAV").exists()

    recordings_a = {
        recording.key: recording.wav_path for recording in load_collection_recordings(collection_a)
    }
    recordings_b = {
        recording.key: recording.wav_path for recording in load_collection_recordings(collection_b)
    }
    assert recordings_a["401_20250211_075558"] == (
        "audio_collection_a/raw_recordings/20250211_075558.WAV"
    )
    assert recordings_b["402_20250212_081000"] == (
        "audio_collection_b/raw_recordings/20250212_081000.WAV"
    )


def test_build_indexes_emits_zero_hum_artifacts_for_raw_only_collections(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collection"
    out_dir = tmp_path / "indexes"
    root.mkdir()

    raw_only = root / "audio_collection_raw"
    (raw_only / "raw_recordings").mkdir(parents=True)
    (raw_only / "raw_recordings" / "401_20250211_075558.WAV").write_bytes(b"WAV")
    write_collection_recordings(
        raw_only,
        [
            _source_recording(
                key="401_20250211_075558",
                collection=raw_only.name,
                wav_path="audio_collection_raw/raw_recordings/401_20250211_075558.WAV",
            )
        ],
    )

    ready = root / "audio_collection_ready"
    (ready / "clips_labelled").mkdir(parents=True)
    (ready / "hums_segmented").mkdir()
    (ready / "raw_recordings").mkdir()
    (ready / "raw_recordings" / "402_20250212_081000.WAV").write_bytes(b"WAV")
    write_collection_recordings(
        ready,
        [
            _source_recording(
                key="402_20250212_081000",
                collection=ready.name,
                wav_path="audio_collection_ready/raw_recordings/402_20250212_081000.WAV",
            )
        ],
    )

    report = build_indexes(
        root=root,
        identity_map=_identity_map(),
        out_dir=out_dir,
        min_source_quality_to_keep=1,
    )

    assert sorted(report.per_collection_payloads.keys()) == [
        "audio_collection_raw",
        "audio_collection_ready",
    ]
    assert report.per_collection_payloads["audio_collection_raw"]["entries"] == []
    assert (out_dir / "audio_collection_raw" / "index.json").is_file()
    assert (out_dir / "audio_collection_ready" / "index.json").is_file()
    assert len(report.merged_payload["recordings"]) == 2


def test_dataset_build_fails_when_target_pool_is_empty(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    merged_index_path = collection_root / "merged_index.json"
    collection_root.mkdir()
    datasets_root.mkdir()
    write_json(
        merged_index_path,
        {
            "meta": {"n_collections": 0},
            "entries": [],
            "recordings": [],
        },
    )

    strategy_config = StrategyConfig.model_validate(
        {
            "target_collection_names": ["audio_collection_ready"],
            "noise_collection_names": ["audio_collection_raw"],
            "split_strategy": "random",
            "seed": 42,
            "min_quality": 2,
            "noise_per_positive": 1.0,
            "noise_mining": {
                "attempts_per_slot": 20,
                "source_category_dirs": ["raw_recordings"],
                "low_quality_as_negative": False,
                "low_quality_threshold": 1,
            },
            "split_fractions": [0.7, 0.15, 0.15],
            "duration_tolerance_s": 0.1,
            "review_gap_s": 0.5,
            "freq_low_hz": 0,
            "freq_high_hz": 4000,
        }
    )

    with pytest.raises(ValueError, match="target pool is empty"):
        build_dataset(
            strategy_name="dataset_zero_target",
            strategy_config=strategy_config,
            collection_root=collection_root,
            merged_index_path=merged_index_path,
            datasets_root=datasets_root,
        )


def test_dataset_build_filters_target_collections_before_positive_selection(
    tmp_path: Path,
) -> None:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    merged_index_path = collection_root / "merged_index.json"
    collection_root.mkdir()
    datasets_root.mkdir()
    write_json(
        merged_index_path,
        {
            "meta": {"n_collections": 1},
            "entries": [
                {
                    "collection": "audio_collection_other",
                    "subject_id": "401",
                    "recording_date": "2025-02-11",
                    "recording_time": "07:55:58",
                    "hum_path": "audio_collection_other/hums_segmented/clip.wav",
                    "hum_start_s": 0.0,
                    "hum_end_s": 1.0,
                    "source_quality": 4,
                    "keep": True,
                    "hum_uid": 1,
                    "source_recording_key": None,
                }
            ],
            "recordings": [],
        },
    )

    strategy_config = StrategyConfig.model_validate(
        {
            "target_collection_names": ["audio_collection_target"],
            "noise_collection_names": ["audio_collection_noise"],
            "split_strategy": "random",
            "seed": 42,
            "min_quality": 2,
            "noise_per_positive": 1.0,
            "noise_mining": {
                "attempts_per_slot": 20,
                "source_category_dirs": ["raw_recordings"],
                "low_quality_as_negative": False,
                "low_quality_threshold": 1,
            },
            "split_fractions": [0.7, 0.15, 0.15],
            "duration_tolerance_s": 0.1,
            "review_gap_s": 0.5,
            "freq_low_hz": 0,
            "freq_high_hz": 4000,
        }
    )

    with pytest.raises(ValueError, match="target pool is empty"):
        build_dataset(
            strategy_name="dataset_filtered_target",
            strategy_config=strategy_config,
            collection_root=collection_root,
            merged_index_path=merged_index_path,
            datasets_root=datasets_root,
        )

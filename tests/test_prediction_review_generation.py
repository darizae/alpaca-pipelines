from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from alpaca_pipelines import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review import (
    CuratedPredictionSourceManifest,
    PredictionReviewSpectrogramConfig,
)
from alpaca_pipelines.prediction.review.curated import (
    CuratedPredictionExportItem,
    _build_curated_example_id,
)


def _build_api(tmp_path: Path) -> PipelineAPI:
    collection_root = tmp_path / "collection"
    datasets_root = tmp_path / "datasets"
    runs_root = tmp_path / "runs"
    collection_root.mkdir()
    datasets_root.mkdir()
    write_json(collection_root / "merged_index.json", {"meta": {}, "entries": []})

    environment = PipelineEnvironment.from_explicit(
        collection_root=collection_root,
        merged_index_path=collection_root / "merged_index.json",
        datasets_root=datasets_root,
        runs_root=runs_root,
    )
    return PipelineAPI(environment)


def _write_test_audio(path: Path) -> None:
    sample_rate = 44_100
    duration_s = 2.0
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def _prepare_completed_prediction_run(
    api: PipelineAPI,
    *,
    audio_file: Path,
    run_type: str = "prediction",
) -> str:
    run_state = api.run_manager.create_run(run_type, {"model_path": "/models/model.pt"})
    api.run_manager.mark_running(run_state.run_id)
    completed = api.run_manager.mark_completed(run_state.run_id)

    prediction_summary_path = (
        Path(completed.run_dir) / "outputs" / "predictions" / "prediction_summary.json"
    )
    write_json(
        prediction_summary_path,
        {
            "run_id": completed.run_id,
            "files": [
                {
                    "audio_file": str(audio_file),
                    "n_windows": 2,
                    "n_detections": 2,
                }
            ],
        },
    )
    return str(completed.run_id)


def test_prediction_review_spectrogram_defaults_and_validation() -> None:
    config = PredictionReviewSpectrogramConfig()

    assert config.window_function == "hann"
    assert config.window_size_samples == 2002
    assert config.hop_size_samples == 1001
    assert config.dft_size == 2048
    assert config.clipping_enabled is False
    assert config.averaging == 1
    assert config.auto_apply is False
    assert config.colormap == "magma"
    assert config.x_axis_label == "Time (s)"
    assert config.y_axis_label == "Frequency (kHz)"
    assert config.show_colorbar is True
    assert config.overlap_percent() == 50.0

    with pytest.raises(ValueError, match="hop_size_samples must be <= window_size_samples"):
        PredictionReviewSpectrogramConfig(window_size_samples=100, hop_size_samples=101)


def test_prediction_review_preview_generates_artifacts_with_renderer_metadata(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)

    manifest_path = tmp_path / "review_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_alpha",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                }
            ],
        },
    )

    payload = api.generate_prediction_review_preview(
        manifest_path=manifest_path,
        item_id="item_001",
    )

    item = payload["item"]
    assert Path(item["spectrogram_png"]).is_file()
    assert Path(item["clip_wav"]).is_file()
    assert Path(item["metadata_json"]).is_file()

    metadata = read_json(Path(item["metadata_json"]))
    assert metadata["spectrogram_config"]["colormap"] == "magma"
    assert metadata["spectrogram_config"]["x_axis_label"] == "Time (s)"
    assert metadata["spectrogram_config"]["y_axis_label"] == "Frequency (kHz)"
    assert metadata["spectrogram_config"]["show_colorbar"] is True
    assert metadata["spectrogram_n_freq_bins"] > 0
    assert metadata["spectrogram_n_frames"] > 0


def test_prediction_review_batch_summary_is_deterministic_and_export_supports_single_and_batch(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)

    manifest_path = tmp_path / "review_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_beta",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                },
                {
                    "item_id": "item_002",
                    "audio_file": str(audio_file),
                    "start_s": 0.9,
                    "end_s": 1.4,
                },
            ],
        },
    )

    first = api.generate_prediction_review_batch(manifest_path=manifest_path)
    second = api.generate_prediction_review_batch(manifest_path=manifest_path)

    assert first == second
    persisted_summary = read_json(Path(first["summary_path"]))
    assert persisted_summary == first

    single_export = api.export_prediction_review_artifacts(
        manifest_path=manifest_path,
        destination_dir=tmp_path / "single_export",
        item_id="item_001",
    )
    assert single_export["n_items"] == 1
    copied_single = single_export["items"][0]["destination"]
    assert Path(copied_single["spectrogram_png"]).is_file()
    assert Path(copied_single["clip_wav"]).is_file()

    batch_export = api.export_prediction_review_artifacts(
        manifest_path=manifest_path,
        destination_dir=tmp_path / "batch_export",
    )
    assert batch_export["n_items"] == 2
    for entry in batch_export["items"]:
        assert Path(entry["destination"]["spectrogram_png"]).is_file()
        assert Path(entry["destination"]["clip_wav"]).is_file()


def test_prediction_review_concat_creates_single_review_wav_without_spectrogram_generation(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)

    manifest_path = tmp_path / "review_manifest_concat.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_concat",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                },
                {
                    "item_id": "item_002",
                    "audio_file": str(audio_file),
                    "start_s": 0.9,
                    "end_s": 1.4,
                },
            ],
        },
    )

    payload = api.concatenate_prediction_review_clips(manifest_path=manifest_path)

    concat_path = Path(payload["concat_wav"])
    assert concat_path.is_file()
    assert payload["n_items"] == 2
    assert payload["items"][0]["item_id"] == "item_001"
    assert payload["items"][1]["item_id"] == "item_002"
    assert payload["items"][0]["begin_time_s"] == pytest.approx(0.0, abs=1e-6)
    assert payload["items"][0]["end_time_s"] == pytest.approx(0.6, abs=1e-6)
    assert payload["items"][1]["begin_time_s"] == pytest.approx(0.6, abs=1e-6)
    assert payload["items"][1]["end_time_s"] == pytest.approx(1.1, abs=1e-6)

    session_dir = concat_path.parent.parent
    assert session_dir.name == "session_concat"
    assert not (session_dir / "items").exists()


def test_prediction_review_concat_accepts_rf_inference_runs(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio_rf.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(
        api,
        audio_file=audio_file,
        run_type="rf_inference",
    )

    manifest_path = tmp_path / "review_manifest_concat_rf.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_concat_rf",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                }
            ],
        },
    )

    payload = api.concatenate_prediction_review_clips(manifest_path=manifest_path)

    assert payload["prediction_run_id"] == run_id
    assert payload["n_items"] == 1
    assert Path(payload["concat_wav"]).is_file()


def test_prediction_review_concat_streams_without_np_concatenate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio_streaming.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)

    manifest_path = tmp_path / "review_manifest_streaming.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_streaming",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.0,
                    "end_s": 0.3,
                },
                {
                    "item_id": "item_002",
                    "audio_file": str(audio_file),
                    "start_s": 0.3,
                    "end_s": 0.6,
                },
            ],
        },
    )

    monkeypatch.setattr(
        np,
        "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not concatenate")),
    )

    payload = api.concatenate_prediction_review_clips(manifest_path=manifest_path)

    assert payload["n_items"] == 2
    assert Path(payload["concat_wav"]).is_file()


def test_prediction_review_flat_snippets_export_writes_manifest_and_flat_wavs(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)
    manifest_path = tmp_path / "review_manifest_flat.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_flat",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "canonical_detection_id": 101,
                    "review_item_id": "review-item-001",
                    "detection_index": 0,
                    "source_collection_name": "audio_collection_a",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "nested/audio.wav",
                    "source_display_path": "audio_collection_a/raw_recordings/nested/audio.wav",
                    "source_recording_key": "a_20250211_075558",
                    "payload_json": {
                        "source_display_path": "audio_collection_a/raw_recordings/nested/audio.wav"
                    },
                },
                {
                    "item_id": "item_002",
                    "audio_file": str(audio_file),
                    "start_s": 0.9,
                    "end_s": 1.4,
                },
            ],
        },
    )
    api.generate_prediction_review_batch(manifest_path=manifest_path)
    payload = api.export_prediction_review_flat_snippets_bundle(manifest_path=manifest_path)
    assert payload["n_items"] == 2
    assert payload["estimated_size_bytes"] > 0
    manifest_payload = read_json(Path(payload["manifest_path"]))
    assert manifest_payload["mode"] == "flat_snippets_bundle"
    assert manifest_payload["items"][0]["snippet_filename"] == "snippet_000000_item_001.wav"
    assert manifest_payload["items"][1]["snippet_filename"] == "snippet_000001_item_002.wav"
    assert manifest_payload["items"][0]["source_collection_name"] == "audio_collection_a"
    assert manifest_payload["items"][0]["canonical_detection_id"] == 101
    assert manifest_payload["items"][0]["review_item_id"] == "review-item-001"
    assert (
        manifest_payload["items"][0]["source_display_path"]
        == "audio_collection_a/raw_recordings/nested/audio.wav"
    )
    assert manifest_payload["items"][0]["source_recording_key"] == "a_20250211_075558"
    assert Path(payload["output_dir"], "snippet_000000_item_001.wav").is_file()


def test_prediction_review_flat_snippets_export_overwrites_existing_targets(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)
    manifest_path = tmp_path / "review_manifest_flat_retry.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_flat_retry",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                }
            ],
        },
    )
    api.generate_prediction_review_batch(manifest_path=manifest_path)
    first = api.export_prediction_review_flat_snippets_bundle(manifest_path=manifest_path)
    snippet_path = Path(first["output_dir"], "snippet_000000_item_001.wav")
    assert snippet_path.is_file()
    snippet_path.write_bytes(b"stale")
    second = api.export_prediction_review_flat_snippets_bundle(manifest_path=manifest_path)
    assert second["n_items"] == 1
    assert snippet_path.is_file()
    assert snippet_path.read_bytes() != b"stale"


def test_materialize_curated_prediction_examples_and_status(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(api, audio_file=audio_file)

    manifest_path = tmp_path / "review_manifest_curated.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_curated",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "source_collection_name": "audio_collection_alpha",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "audio.wav",
                    "source_recording_key": "401_20250211_075558",
                }
            ],
        },
    )
    labels_path = tmp_path / "labels.json"
    write_json(
        labels_path,
        {
            "schema_version": 1,
            "labels": {
                "item_001": "target",
            },
        },
    )

    payload = api.materialize_curated_prediction_examples(
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert payload["counts_by_label"] == {"target": 1, "noise": 0}
    assert payload["category_names"] == ["hums_curated_manual_review"]
    assert payload["created_count"] == 1
    assert payload["updated_count"] == 0
    assert payload["skipped_count"] == 0

    manifest_paths = payload["manifest_paths"]
    assert len(manifest_paths) == 1
    curated_manifest = read_json(Path(manifest_paths[0]))
    CuratedPredictionSourceManifest.model_validate(curated_manifest)
    assert curated_manifest["source_type"] == "manual_review_curated"
    assert curated_manifest["items"][0]["label"] == "target"
    assert Path(curated_manifest["items"][0]["snippet_wav_path"]).is_file()
    assert "/hums_curated_manual_review/" in curated_manifest["items"][0]["snippet_wav_path"]

    second_payload = api.materialize_curated_prediction_examples(
        manifest_path=manifest_path,
        labels_path=labels_path,
    )
    assert second_payload["created_count"] == 0
    assert second_payload["updated_count"] == 0
    assert second_payload["skipped_count"] == 1

    status = api.list_curated_prediction_categories()
    assert status["counts_by_label"]["target"] == 1
    assert status["counts_by_provenance_type"]["manual_review_curated"] == 1
    assert "hums_curated_manual_review" in status["category_names"]


def test_materialize_curated_prediction_examples_accepts_rf_inference_runs(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    audio_file = tmp_path / "audio_curated_rf.wav"
    _write_test_audio(audio_file)
    run_id = _prepare_completed_prediction_run(
        api,
        audio_file=audio_file,
        run_type="rf_inference",
    )

    manifest_path = tmp_path / "review_manifest_curated_rf.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_curated_rf",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(audio_file),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "source_collection_name": "audio_collection_alpha",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "audio_curated_rf.wav",
                    "source_recording_key": "401_20250211_075558",
                }
            ],
        },
    )
    labels_path = tmp_path / "labels_rf.json"
    write_json(
        labels_path,
        {
            "schema_version": 1,
            "labels": {
                "item_001": "noise",
            },
        },
    )

    payload = api.materialize_curated_prediction_examples(
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert payload["prediction_run_id"] == run_id
    assert payload["counts_by_label"] == {"target": 0, "noise": 1}


def test_materialize_curated_prediction_examples_accepts_curated_export_manifest(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    collection_audio = (
        tmp_path / "collection" / "audio_collection_alpha" / "raw_recordings" / "audio.wav"
    )
    _write_test_audio(collection_audio)
    run_id = _prepare_completed_prediction_run(api, audio_file=collection_audio)

    export_manifest_path = tmp_path / "curated_export_manifest.json"
    write_json(
        export_manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "review_session_id": "session_curated",
            "items": [
                {
                    "curated_example_id": "ui-curated-example-id",
                    "review_item_id": "item_001",
                    "source_audio_file": str(collection_audio),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "label": "target",
                    "source_collection_name": "audio_collection_alpha",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "audio.wav",
                    "source_recording_key": "401_20250211_075558",
                }
            ],
        },
    )

    payload = api.materialize_curated_prediction_examples(
        curated_export_manifest=export_manifest_path
    )

    assert payload["counts_by_label"] == {"target": 1, "noise": 0}
    curated_manifest = read_json(Path(payload["manifest_paths"][0]))
    assert curated_manifest["source_relative_path"] == "audio.wav"
    assert (
        curated_manifest["source_display_path"] == "audio_collection_alpha/raw_recordings/audio.wav"
    )
    assert (
        curated_manifest["items"][0]["source_display_path"]
        == "audio_collection_alpha/raw_recordings/audio.wav"
    )
    assert curated_manifest["items"][0]["curated_example_id"] == "ui-curated-example-id"


def test_curated_prediction_export_item_accepts_curated_example_id() -> None:
    item = CuratedPredictionExportItem.model_validate(
        {
            "curated_example_id": "ui-curated-example-id",
            "review_item_id": "item_001",
            "source_audio_file": "/tmp/audio.wav",
            "start_s": 0.1,
            "end_s": 0.7,
            "label": "target",
        }
    )
    assert item.curated_example_id == "ui-curated-example-id"


def test_materialize_curated_prediction_examples_falls_back_to_hash_id_when_absent(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    collection_audio = (
        tmp_path / "collection" / "audio_collection_alpha" / "raw_recordings" / "audio.wav"
    )
    _write_test_audio(collection_audio)
    run_id = _prepare_completed_prediction_run(api, audio_file=collection_audio)

    export_manifest_path = tmp_path / "curated_export_manifest_no_id.json"
    write_json(
        export_manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "review_session_id": "session_curated",
            "items": [
                {
                    "review_item_id": "item_001",
                    "source_audio_file": str(collection_audio),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "label": "target",
                    "source_collection_name": "audio_collection_alpha",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "audio.wav",
                    "source_recording_key": "401_20250211_075558",
                }
            ],
        },
    )

    payload = api.materialize_curated_prediction_examples(
        curated_export_manifest=export_manifest_path
    )
    curated_manifest = read_json(Path(payload["manifest_paths"][0]))
    assert curated_manifest["items"][0]["curated_example_id"] == _build_curated_example_id(
        prediction_run_id=run_id,
        review_session_id="session_curated",
        review_item_id="item_001",
    )


def test_materialize_curated_prediction_examples_rejects_collection_prefixed_relative_path(
    tmp_path: Path,
) -> None:
    api = _build_api(tmp_path)
    collection_audio = (
        tmp_path / "collection" / "audio_collection_alpha" / "raw_recordings" / "audio.wav"
    )
    _write_test_audio(collection_audio)
    run_id = _prepare_completed_prediction_run(api, audio_file=collection_audio)

    manifest_path = tmp_path / "review_manifest_curated.json"
    labels_path = tmp_path / "labels.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "prediction_run_id": run_id,
            "session_id": "session_curated",
            "items": [
                {
                    "item_id": "item_001",
                    "audio_file": str(collection_audio),
                    "start_s": 0.1,
                    "end_s": 0.7,
                    "source_collection_name": "audio_collection_alpha",
                    "source_category_dir": "raw_recordings",
                    "source_relative_path": "audio_collection_alpha/raw_recordings/audio.wav",
                    "source_recording_key": "401_20250211_075558",
                }
            ],
        },
    )
    write_json(labels_path, {"schema_version": 1, "labels": {"item_001": "target"}})

    with pytest.raises(ValueError, match="source_relative_path"):
        api.materialize_curated_prediction_examples(
            manifest_path=manifest_path,
            labels_path=labels_path,
        )

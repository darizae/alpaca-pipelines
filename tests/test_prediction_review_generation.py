from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from alpaca_pipelines import PipelineAPI
from alpaca_pipelines.config import PipelineEnvironment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review import PredictionReviewSpectrogramConfig


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
) -> str:
    run_state = api.run_manager.create_run("prediction", {"model_path": "/models/model.pt"})
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

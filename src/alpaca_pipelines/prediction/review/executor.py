"""Prediction manual-review artifact generation and export."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from alpaca_pipelines.contracts import RunState
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review.config import (
    PredictionReviewSessionItem,
    PredictionReviewSessionManifest,
    PredictionReviewSpectrogramConfig,
)
from alpaca_pipelines.runs.manager import RunManager

matplotlib.use("Agg")

_MANUAL_REVIEW_DIR = "manual_review"
_SESSION_SUMMARY_FILENAME = "summary.json"
_RAVEN_DIR = "raven"
_RAVEN_CONCAT_FILENAME = "review_concat.wav"
_FLAT_SNIPPETS_DIR = "flat_snippets_bundle"
_FLAT_SNIPPETS_MANIFEST_FILENAME = "snippets_manifest.json"
_REVIEW_ELIGIBLE_RUN_TYPES = {"prediction", "rf_inference"}


def generate_prediction_review_preview(
    *,
    run_manager: RunManager,
    manifest_path: Path,
    item_id: str,
    spectrogram_config: PredictionReviewSpectrogramConfig | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    item = _resolve_item(manifest, item_id)
    config = _resolve_spectrogram_config(manifest, spectrogram_config)
    run_state = _resolve_prediction_run(run_manager, manifest.prediction_run_id)
    _validate_manifest_audio_inventory(run_state, manifest)

    session_dir = _session_dir(run_state, manifest.session_id)
    item_summary = _generate_item_artifacts(
        session_dir=session_dir,
        item=item,
        config=config,
    )

    payload = {
        "mode": "preview",
        "prediction_run_id": manifest.prediction_run_id,
        "session_id": manifest.session_id,
        "item_id": item_id,
        "spectrogram_config": _spectrogram_config_payload(config),
        "item": item_summary,
    }

    preview_summary_path = session_dir / "preview_{}.json".format(item_id)
    payload["summary_path"] = str(preview_summary_path)
    write_json(preview_summary_path, payload)
    return payload


def generate_prediction_review_batch(
    *,
    run_manager: RunManager,
    manifest_path: Path,
    spectrogram_config: PredictionReviewSpectrogramConfig | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    config = _resolve_spectrogram_config(manifest, spectrogram_config)
    run_state = _resolve_prediction_run(run_manager, manifest.prediction_run_id)
    _validate_manifest_audio_inventory(run_state, manifest)

    session_dir = _session_dir(run_state, manifest.session_id)

    item_summaries: list[dict[str, Any]] = []
    for item in manifest.items:
        item_summaries.append(
            _generate_item_artifacts(
                session_dir=session_dir,
                item=item,
                config=config,
            )
        )

    summary_path = session_dir / _SESSION_SUMMARY_FILENAME
    payload = {
        "mode": "batch",
        "prediction_run_id": manifest.prediction_run_id,
        "session_id": manifest.session_id,
        "session_dir": str(session_dir),
        "summary_path": str(summary_path),
        "spectrogram_config": _spectrogram_config_payload(config),
        "n_items": len(item_summaries),
        "items": item_summaries,
    }
    write_json(summary_path, payload)
    return payload


def concatenate_prediction_review_clips(
    *,
    run_manager: RunManager,
    manifest_path: Path,
    output_wav: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    run_state = _resolve_prediction_run(run_manager, manifest.prediction_run_id)
    _validate_manifest_audio_inventory(run_state, manifest)

    session_dir = _session_dir(run_state, manifest.session_id)
    output_path = (
        output_wav if output_wav is not None else session_dir / _RAVEN_DIR / _RAVEN_CONCAT_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    segments: list[np.ndarray] = []
    offsets: list[dict[str, float | str]] = []
    samplerate: int | None = None
    current_offset = 0.0
    for item in manifest.items:
        cached_audio = audio_cache.get(item.audio_file)
        if cached_audio is None:
            cached_audio = _load_mono_audio(Path(item.audio_file))
            audio_cache[item.audio_file] = cached_audio
        mono_audio, item_samplerate = cached_audio
        if samplerate is None:
            samplerate = item_samplerate
        elif item_samplerate != samplerate:
            raise ValueError(
                "All prediction review clips must have the same sample rate for concatenation"
            )
        clip, _, _ = _extract_item_clip(
            item=item,
            mono_audio=mono_audio,
            sample_rate=item_samplerate,
        )
        clip_duration_s = float(clip.shape[0]) / float(item_samplerate)
        begin_time_s = current_offset
        end_time_s = begin_time_s + clip_duration_s
        offsets.append(
            {
                "item_id": item.item_id,
                "begin_time_s": begin_time_s,
                "end_time_s": end_time_s,
            }
        )
        segments.append(clip)
        current_offset = end_time_s

    if samplerate is None or not segments:
        raise ValueError("Prediction review manifest has no clips to concatenate")
    concatenated = np.concatenate(segments)
    sf.write(output_path, concatenated.astype(np.float32), samplerate)

    return {
        "prediction_run_id": manifest.prediction_run_id,
        "session_id": manifest.session_id,
        "concat_wav": str(output_path),
        "n_items": len(offsets),
        "items": offsets,
    }


def export_prediction_review_artifacts(
    *,
    run_manager: RunManager,
    manifest_path: Path,
    destination_dir: Path,
    item_id: str | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    run_state = _resolve_prediction_run(run_manager, manifest.prediction_run_id)
    _validate_manifest_audio_inventory(run_state, manifest)

    session_dir = _session_dir(run_state, manifest.session_id)

    selected_items: list[PredictionReviewSessionItem]
    if item_id is None:
        selected_items = manifest.items
    else:
        selected_items = [_resolve_item(manifest, item_id)]

    destination_session_dir = destination_dir / manifest.session_id
    destination_session_dir.mkdir(parents=True, exist_ok=True)

    copied_items: list[dict[str, Any]] = []
    for item in selected_items:
        source_item_dir = session_dir / "items" / item.item_id
        source_spectrogram = source_item_dir / "spectrogram.png"
        source_clip = source_item_dir / "clip.wav"
        source_metadata = source_item_dir / "artifact_metadata.json"
        for artifact_path in (source_spectrogram, source_clip, source_metadata):
            if not artifact_path.is_file():
                raise FileNotFoundError("Missing review artifact: {}".format(artifact_path))

        destination_item_dir = destination_session_dir / item.item_id
        destination_item_dir.mkdir(parents=True, exist_ok=True)

        destination_spectrogram = destination_item_dir / source_spectrogram.name
        destination_clip = destination_item_dir / source_clip.name
        destination_metadata = destination_item_dir / source_metadata.name

        for destination_path in (
            destination_spectrogram,
            destination_clip,
            destination_metadata,
        ):
            if destination_path.exists():
                raise FileExistsError("Export target already exists: {}".format(destination_path))

        shutil.copy2(source_spectrogram, destination_spectrogram)
        shutil.copy2(source_clip, destination_clip)
        shutil.copy2(source_metadata, destination_metadata)

        copied_items.append(
            {
                "item_id": item.item_id,
                "canonical_detection_id": item.canonical_detection_id,
                "review_item_id": item.review_item_id,
                "detection_index": item.detection_index,
                "source_display_path": item.source_display_path,
                "source": {
                    "spectrogram_png": str(source_spectrogram),
                    "clip_wav": str(source_clip),
                    "metadata_json": str(source_metadata),
                },
                "destination": {
                    "spectrogram_png": str(destination_spectrogram),
                    "clip_wav": str(destination_clip),
                    "metadata_json": str(destination_metadata),
                },
            }
        )

    export_summary_path = destination_session_dir / "export_summary.json"
    payload = {
        "prediction_run_id": manifest.prediction_run_id,
        "session_id": manifest.session_id,
        "destination_dir": str(destination_session_dir),
        "summary_path": str(export_summary_path),
        "n_items": len(copied_items),
        "items": copied_items,
    }
    write_json(export_summary_path, payload)
    return payload


def export_prediction_review_flat_snippets_bundle(
    *,
    run_manager: RunManager,
    manifest_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    run_state = _resolve_prediction_run(run_manager, manifest.prediction_run_id)
    _validate_manifest_audio_inventory(run_state, manifest)

    session_dir = _session_dir(run_state, manifest.session_id)
    package_dir = output_dir if output_dir is not None else session_dir / _FLAT_SNIPPETS_DIR
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, Any]] = []
    total_clip_bytes = 0
    for item_index, item in enumerate(manifest.items):
        source_clip = session_dir / "items" / item.item_id / "clip.wav"
        if not source_clip.is_file():
            raise FileNotFoundError("Missing review clip: {}".format(source_clip))
        clip_filename = "snippet_{:06d}_{}.wav".format(item_index, item.item_id)
        destination_clip = package_dir / clip_filename
        if destination_clip.exists() and destination_clip.is_dir():
            raise FileExistsError("Export target already exists: {}".format(destination_clip))
        if destination_clip.exists():
            destination_clip.unlink()
        shutil.copy2(source_clip, destination_clip)
        clip_size_bytes = destination_clip.stat().st_size
        total_clip_bytes += clip_size_bytes
        source_display_path = item.source_display_path
        if source_display_path is None and isinstance(item.payload_json, dict):
            payload_display_path = item.payload_json.get("source_display_path")
            if isinstance(payload_display_path, str):
                source_display_path = payload_display_path
        manifest_items.append(
            {
                "item_index": item_index,
                "item_id": item.item_id,
                "snippet_filename": clip_filename,
                "snippet_size_bytes": clip_size_bytes,
                "audio_file": item.audio_file,
                "start_s": item.start_s,
                "end_s": item.end_s,
                "detection_score": item.detection_score,
                "canonical_detection_id": item.canonical_detection_id,
                "detection_index": item.detection_index,
                "review_item_id": item.review_item_id,
                "source_collection_name": item.source_collection_name,
                "source_category_dir": item.source_category_dir,
                "source_relative_path": item.source_relative_path,
                "source_display_path": source_display_path,
                "source_recording_key": item.source_recording_key,
            }
        )

    summary_path = package_dir / "summary.json"
    manifest_payload = {
        "schema_version": 1,
        "mode": "flat_snippets_bundle",
        "prediction_run_id": manifest.prediction_run_id,
        "session_id": manifest.session_id,
        "output_dir": str(package_dir),
        "summary_path": str(summary_path),
        "manifest_path": str(package_dir / _FLAT_SNIPPETS_MANIFEST_FILENAME),
        "n_items": len(manifest_items),
        "estimated_size_bytes": total_clip_bytes,
        "items": manifest_items,
    }
    write_json(package_dir / _FLAT_SNIPPETS_MANIFEST_FILENAME, manifest_payload)
    write_json(summary_path, manifest_payload)
    return manifest_payload


def _load_manifest(manifest_path: Path) -> PredictionReviewSessionManifest:
    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest file not found: {}".format(manifest_path))
    manifest_raw = read_json(manifest_path)
    if not isinstance(manifest_raw, dict):
        raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
    manifest = PredictionReviewSessionManifest.model_validate(manifest_raw)

    if not _is_safe_path_segment(manifest.session_id):
        raise ValueError("session_id must be a safe path segment: {}".format(manifest.session_id))
    for item in manifest.items:
        if not _is_safe_path_segment(item.item_id):
            raise ValueError("item_id must be a safe path segment: {}".format(item.item_id))

    return manifest


def _resolve_spectrogram_config(
    manifest: PredictionReviewSessionManifest,
    spectrogram_config: PredictionReviewSpectrogramConfig | None,
) -> PredictionReviewSpectrogramConfig:
    if spectrogram_config is not None:
        return spectrogram_config
    if manifest.spectrogram_config is not None:
        return manifest.spectrogram_config
    return PredictionReviewSpectrogramConfig()


def _resolve_prediction_run(run_manager: RunManager, run_id: str) -> RunState:
    run_state = run_manager.find_run(run_id)
    if run_state.run_type not in _REVIEW_ELIGIBLE_RUN_TYPES:
        raise ValueError(
            "Expected inference run (prediction or rf_inference), got: {}".format(
                run_state.run_type
            )
        )
    if run_state.status != "completed":
        raise ValueError(
            "Inference run must be completed for review generation, status: {}".format(
                run_state.status
            )
        )
    return run_state


def _resolve_item(
    manifest: PredictionReviewSessionManifest,
    item_id: str,
) -> PredictionReviewSessionItem:
    for item in manifest.items:
        if item.item_id == item_id:
            return item
    raise ValueError("item_id not found in manifest: {}".format(item_id))


def _prediction_audio_inventory(run_state: RunState) -> set[str]:
    run_dir = Path(run_state.run_dir)
    summary_path = run_dir / "outputs" / "predictions" / "prediction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Prediction summary not found: {}".format(summary_path))

    summary_raw = read_json(summary_path)
    if not isinstance(summary_raw, dict):
        raise ValueError("Expected JSON object: {}".format(summary_path))

    files = summary_raw.get("files")
    if not isinstance(files, list):
        raise ValueError("prediction_summary.json missing 'files' list: {}".format(summary_path))

    inventory: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid prediction summary entry: {}".format(entry))
        audio_file = entry.get("audio_file")
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError("Invalid prediction summary audio_file entry: {}".format(entry))
        inventory.add(audio_file)

    return inventory


def _validate_manifest_audio_inventory(
    run_state: RunState,
    manifest: PredictionReviewSessionManifest,
) -> None:
    inventory = _prediction_audio_inventory(run_state)
    for item in manifest.items:
        if item.audio_file not in inventory:
            raise ValueError(
                "Manifest audio_file is not part of prediction run {}: {}".format(
                    manifest.prediction_run_id,
                    item.audio_file,
                )
            )


def _session_dir(run_state: RunState, session_id: str) -> Path:
    session_dir = Path(run_state.run_dir) / "outputs" / _MANUAL_REVIEW_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _generate_item_artifacts(
    *,
    session_dir: Path,
    item: PredictionReviewSessionItem,
    config: PredictionReviewSpectrogramConfig,
) -> dict[str, Any]:
    mono_audio, sample_rate = _load_mono_audio(Path(item.audio_file))
    clip, start_sample, end_sample = _extract_item_clip(
        item=item,
        mono_audio=mono_audio,
        sample_rate=sample_rate,
    )

    spectrogram_db, freqs_hz, times_s = _compute_spectrogram(
        clip=clip,
        sample_rate=sample_rate,
        config=config,
    )

    item_dir = session_dir / "items" / item.item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    clip_path = item_dir / "clip.wav"
    spectrogram_path = item_dir / "spectrogram.png"
    metadata_path = item_dir / "artifact_metadata.json"

    sf.write(clip_path, clip.astype(np.float32), sample_rate)
    _render_spectrogram(
        spectrogram_db=spectrogram_db,
        freqs_hz=freqs_hz,
        times_s=times_s,
        output_path=spectrogram_path,
        config=config,
    )

    metadata = {
        "item_id": item.item_id,
        "audio_file": item.audio_file,
        "start_s": item.start_s,
        "end_s": item.end_s,
        "sample_rate": sample_rate,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "clip_n_samples": int(clip.shape[0]),
        "clip_duration_s": round(float(clip.shape[0]) / float(sample_rate), 6),
        "spectrogram_n_freq_bins": int(spectrogram_db.shape[0]),
        "spectrogram_n_frames": int(spectrogram_db.shape[1]),
        "spectrogram_freq_min_khz": float(freqs_hz[0] / 1000.0),
        "spectrogram_freq_max_khz": float(freqs_hz[-1] / 1000.0),
        "spectrogram_time_min_s": float(times_s[0]),
        "spectrogram_time_max_s": float(times_s[-1]),
        "spectrogram_config": _spectrogram_config_payload(config),
    }
    write_json(metadata_path, metadata)

    return {
        "item_id": item.item_id,
        "audio_file": item.audio_file,
        "start_s": item.start_s,
        "end_s": item.end_s,
        "spectrogram_png": str(spectrogram_path),
        "clip_wav": str(clip_path),
        "metadata_json": str(metadata_path),
        "relative_spectrogram_png": str(spectrogram_path.relative_to(session_dir)),
        "relative_clip_wav": str(clip_path.relative_to(session_dir)),
        "relative_metadata_json": str(metadata_path.relative_to(session_dir)),
        "spectrogram_shape": [int(spectrogram_db.shape[0]), int(spectrogram_db.shape[1])],
        "clip_duration_s": metadata["clip_duration_s"],
    }


def _compute_spectrogram(
    *,
    clip: np.ndarray,
    sample_rate: int,
    config: PredictionReviewSpectrogramConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = torch.from_numpy(clip).float().unsqueeze(0)
    window = torch.hann_window(config.window_size_samples)

    stft = torch.stft(
        input=signal,
        n_fft=config.dft_size,
        hop_length=config.hop_size_samples,
        win_length=config.window_size_samples,
        window=window,
        center=True,
        onesided=True,
        return_complex=True,
    )

    power = stft.abs().pow(2).squeeze(0).cpu().numpy()
    if power.ndim != 2 or power.shape[1] == 0:
        raise ValueError("Unable to compute spectrogram for clip")

    spectrogram_db = 10.0 * np.log10(np.maximum(power, 1e-12))

    if config.averaging > 1:
        if spectrogram_db.shape[1] < config.averaging:
            raise ValueError(
                "averaging={} requires at least {} spectrogram frames; got {}".format(
                    config.averaging,
                    config.averaging,
                    spectrogram_db.shape[1],
                )
            )
        kernel = np.ones(config.averaging, dtype=np.float32) / float(config.averaging)
        spectrogram_db = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis=1,
            arr=spectrogram_db,
        )

    if config.clipping_enabled:
        spectrogram_db = np.clip(
            spectrogram_db,
            config.clipping_min_db,
            config.clipping_max_db,
        )

    frequencies_hz = np.linspace(
        0.0,
        float(sample_rate) / 2.0,
        num=spectrogram_db.shape[0],
        dtype=np.float32,
    )
    time_step_s = float(config.hop_size_samples) / float(sample_rate)
    times_s = np.arange(spectrogram_db.shape[1], dtype=np.float32) * time_step_s

    return spectrogram_db, frequencies_hz, times_s


def _render_spectrogram(
    *,
    spectrogram_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    output_path: Path,
    config: PredictionReviewSpectrogramConfig,
) -> None:
    freqs_khz = freqs_hz / 1000.0

    fig, ax = plt.subplots(figsize=(9, 4))
    mesh = ax.pcolormesh(
        times_s,
        freqs_khz,
        spectrogram_db,
        shading="auto",
        cmap=config.colormap,
    )

    if config.show_axes:
        ax.set_xlabel(config.x_axis_label)
        ax.set_ylabel(config.y_axis_label)
    else:
        ax.set_axis_off()

    if config.show_colorbar:
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _spectrogram_config_payload(config: PredictionReviewSpectrogramConfig) -> dict[str, Any]:
    payload = config.model_dump()
    payload["overlap_percent"] = config.overlap_percent()
    return payload


def _load_mono_audio(audio_file: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_file, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError("Audio file is empty: {}".format(audio_file))
    mono_audio = np.mean(audio, axis=1)
    return mono_audio, int(sample_rate)


def _extract_item_clip(
    *,
    item: PredictionReviewSessionItem,
    mono_audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, int, int]:
    total_samples = int(mono_audio.shape[0])
    start_sample = int(round(item.start_s * sample_rate))
    end_sample = int(round(item.end_s * sample_rate))

    if start_sample < 0:
        raise ValueError("start_s resolves to negative sample index: {}".format(item.start_s))
    if end_sample > total_samples:
        raise ValueError(
            "end_s resolves outside audio bounds: end_s={} total_samples={} sample_rate={}".format(
                item.end_s,
                total_samples,
                sample_rate,
            )
        )
    if end_sample <= start_sample:
        raise ValueError(
            "Invalid clip bounds for item {}: start_s={} end_s={}".format(
                item.item_id,
                item.start_s,
                item.end_s,
            )
        )

    clip = mono_audio[start_sample:end_sample]
    if clip.size == 0:
        raise ValueError("Generated clip is empty for item: {}".format(item.item_id))
    return clip, start_sample, end_sample


def _is_safe_path_segment(value: str) -> bool:
    if not value or value in {".", ".."}:
        return False
    return "/" not in value and "\\" not in value and ".." not in value

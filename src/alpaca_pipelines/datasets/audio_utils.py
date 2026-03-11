from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem


def wav_duration_seconds(path: Path, fs: FileSystem = _DEFAULT_FS) -> float:
    try:
        fh = fs.open_read(path)
        try:
            info = sf.info(fh)
        finally:
            fh.close()
    except Exception as exc:
        raise ValueError(f"Failed to read wav header: {path}") from exc

    if info.samplerate <= 0:
        raise ValueError(f"Invalid samplerate in wav header: {path} (samplerate={info.samplerate})")
    return float(info.frames) / float(info.samplerate)


def wav_samplerate(path: Path, fs: FileSystem = _DEFAULT_FS) -> int:
    try:
        fh = fs.open_read(path)
        try:
            info = sf.info(fh)
        finally:
            fh.close()
    except Exception as exc:
        raise ValueError(f"Failed to read wav header: {path}") from exc
    return int(info.samplerate)


def validate_snippet_duration(
    wav_path: Path,
    expected_start_s: float,
    expected_end_s: float,
    tolerance_s: float,
    fs: FileSystem = _DEFAULT_FS,
) -> float:
    actual_duration = wav_duration_seconds(wav_path, fs)
    expected_duration = expected_end_s - expected_start_s
    mismatch = abs(actual_duration - expected_duration)
    if mismatch > tolerance_s:
        raise ValueError(
            f"Duration mismatch for {wav_path}: "
            f"actual={actual_duration:.4f}s, "
            f"expected={expected_duration:.4f}s "
            f"(from {expected_start_s}-{expected_end_s}), "
            f"mismatch={mismatch:.4f}s exceeds tolerance={tolerance_s}s"
        )
    return actual_duration


def extract_segment(
    source_path: Path,
    start_s: float,
    end_s: float,
    destination_path: Path,
    duration_tolerance_s: float = 0.05,
    fs: FileSystem = _DEFAULT_FS,
) -> float:
    try:
        source_handle = fs.open_read(source_path)
        try:
            info = sf.info(source_handle)
            samplerate = int(info.samplerate)

            start_frame = int(start_s * samplerate)
            stop_frame = int(end_s * samplerate)

            source_handle.seek(0)
            audio_data, read_samplerate = sf.read(
                source_handle,
                start=start_frame,
                stop=stop_frame,
                dtype="float32",
            )
        finally:
            source_handle.close()
    except Exception as exc:
        raise ValueError(f"Failed to read source audio segment: {source_path}") from exc

    if int(read_samplerate) != samplerate:
        raise ValueError(
            "Samplerate mismatch while reading {}: read={}, info={}".format(
                source_path, read_samplerate, samplerate
            )
        )

    actual_duration = float(len(audio_data)) / float(samplerate)
    expected_duration = end_s - start_s
    if abs(actual_duration - expected_duration) > duration_tolerance_s:
        raise ValueError(
            f"Extracted segment duration mismatch for {source_path}: "
            f"actual={actual_duration:.4f}s, expected={expected_duration:.4f}s "
            f"(from {start_s}-{end_s})"
        )

    fs.makedirs(destination_path.parent)

    try:
        out_handle = fs.open_write(destination_path)
        try:
            sf.write(out_handle, audio_data, samplerate)
        finally:
            out_handle.close()
    except Exception as exc:
        raise ValueError(f"Failed to write extracted segment: {destination_path}") from exc

    return actual_duration


def concatenate_wavs(
    wav_paths: list[Path],
    output_path: Path,
    gap_seconds: float,
    target_samplerate: int | None = None,
    fs: FileSystem = _DEFAULT_FS,
) -> list[tuple[float, float]]:
    if not wav_paths:
        raise ValueError("No wav files to concatenate")

    if target_samplerate is None:
        target_samplerate = wav_samplerate(wav_paths[0], fs)

    gap_samples = int(gap_seconds * target_samplerate)
    silence_gap = np.zeros(gap_samples, dtype=np.float32)

    segments: list[np.ndarray] = []
    offsets: list[tuple[float, float]] = []
    current_offset = 0.0

    for idx, wav_path in enumerate(wav_paths):
        try:
            fh = fs.open_read(wav_path)
            try:
                audio_data, samplerate = sf.read(fh, dtype="float32")
            finally:
                fh.close()
        except Exception as exc:
            raise ValueError(f"Failed to read wav for concatenation: {wav_path}") from exc

        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        if int(samplerate) != int(target_samplerate):
            raise ValueError(
                f"Samplerate mismatch: {wav_path} has {samplerate}, expected {target_samplerate}"
            )

        duration = float(len(audio_data)) / float(samplerate)
        offsets.append((current_offset, current_offset + duration))
        segments.append(audio_data)

        current_offset += duration

        if idx < len(wav_paths) - 1:
            segments.append(silence_gap)
            current_offset += gap_seconds

    concatenated = np.concatenate(segments)

    try:
        out_handle = fs.open_write(output_path)
        try:
            sf.write(out_handle, concatenated, int(target_samplerate))
        finally:
            out_handle.close()
    except Exception as exc:
        raise ValueError(f"Failed to write concatenated wav: {output_path}") from exc

    return offsets

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from alpaca_pipelines.collections.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.collections.io_utils import read_json, write_json

RAW_RECORDINGS_DIR = "raw_recordings"
COLLECTION_RECORDINGS_FILENAME = "recordings.json"

_TIMESTAMP_STEM_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})$")
_SETTING_LINE_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<value>.+?),?$")


class RecorderTrackPoint(BaseModel):
    pps_number: int
    audiomoth_time: str
    samples: int
    total_samples: int
    timer_count: int
    buffers_filled: int
    buffers_written: int
    last_rmc_audiomoth_time: str | None = None
    last_rmc_gps_time: str | None = None
    gps_status: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class SourceRecording(BaseModel):
    key: str
    collection: str
    subject_id: str
    deployment_token: str | None = None
    wav_path: str
    csv_path: str | None = None
    device_path: str | None = None
    settings_path: str | None = None
    log_path: str | None = None
    device_id: str | None = None
    firmware_version: str | None = None
    firmware_description: str | None = None
    settings: dict[str, Any] | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    total_samples: int | None = None
    track_points: list[RecorderTrackPoint] | None = None


class CollectionRecordings(BaseModel):
    recordings: list[SourceRecording] = Field(default_factory=list)


def derive_source_recording_key(
    subject_id: str,
    recording_date: str,
    recording_time: str,
) -> str:
    compact_date = recording_date.replace("-", "")
    compact_time = recording_time.replace(":", "")
    return f"{subject_id}_{compact_date}_{compact_time}"


def derive_source_recording_key_from_stem(subject_id: str, stem: str) -> str:
    match = _TIMESTAMP_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Unsupported recording stem: {stem}")
    return f"{subject_id}_{match['date']}_{match['time']}"


def stem_to_iso_timestamp(stem: str) -> str:
    match = _TIMESTAMP_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Unsupported recording stem: {stem}")
    return (
        f"{match['date'][0:4]}-{match['date'][4:6]}-{match['date'][6:8]}"
        f"T{match['time'][0:2]}:{match['time'][2:4]}:{match['time'][4:6]}"
    )


def compute_recording_counts(
    recordings: list[SourceRecording],
) -> tuple[int, int]:
    return (
        len(recordings),
        sum(1 for recording in recordings if recording.track_points),
    )


def maybe_recording_bounds_from_duration(
    start_time: str | None,
    duration_seconds: float | None,
) -> str | None:
    if start_time is None or duration_seconds is None:
        return None
    try:
        started_at = datetime.fromisoformat(start_time)
    except ValueError:
        return None
    ended_at = started_at + timedelta(seconds=duration_seconds)
    return ended_at.isoformat(timespec="milliseconds")


def load_collection_recordings(
    collection_dir: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> list[SourceRecording]:
    path = collection_dir / COLLECTION_RECORDINGS_FILENAME
    if not fs.exists(path):
        return []
    payload = read_json(path, fs)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return CollectionRecordings.model_validate(payload).recordings


def write_collection_recordings(
    collection_dir: Path,
    recordings: list[SourceRecording],
    fs: FileSystem = _DEFAULT_FS,
) -> Path:
    path = collection_dir / COLLECTION_RECORDINGS_FILENAME
    payload = CollectionRecordings(recordings=recordings).model_dump()
    write_json(path, payload, fs)
    return path


def parse_audiomoth_csv(path: Path, fs: FileSystem = _DEFAULT_FS) -> list[RecorderTrackPoint]:
    content = fs.read_text(path)
    reader = csv.DictReader(content.splitlines())
    points: list[RecorderTrackPoint] = []
    for row in reader:
        gps_status = _optional_text(row.get("STATUS"))
        latitude = _decimal_coordinate(
            row.get("LAT_DEG"),
            row.get("LAT_MIN"),
            row.get("LAT_DIR"),
        )
        longitude = _decimal_coordinate(
            row.get("LONG_DEG"),
            row.get("LONG_MIN"),
            row.get("LONG_DIR"),
        )
        points.append(
            RecorderTrackPoint(
                pps_number=_int_value(row.get("PPS_NUMBER")),
                audiomoth_time=_required_text(row.get("AUDIOMOTH_TIME"), "AUDIOMOTH_TIME"),
                samples=_int_value(row.get("SAMPLES")),
                total_samples=_int_value(row.get("TOTAL_SAMPLES")),
                timer_count=_int_value(row.get("TIMER_COUNT")),
                buffers_filled=_int_value(row.get("BUFFERS_FILLED")),
                buffers_written=_int_value(row.get("BUFFERS_WRITTEN")),
                last_rmc_audiomoth_time=_optional_text(row.get("LAST_RMC_AUDIOMOTH_TIME")),
                last_rmc_gps_time=_optional_text(row.get("LAST_RMC_GPS_TIME")),
                gps_status=gps_status,
                latitude=latitude,
                longitude=longitude,
            )
        )
    return points


def parse_device_txt(path: Path, fs: FileSystem = _DEFAULT_FS) -> dict[str, str | None]:
    device_id: str | None = None
    firmware_version: str | None = None
    firmware_description: str | None = None
    for raw_line in fs.read_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith("Device:"):
            device_id = line.split(":", 1)[1].strip() or None
        elif line.startswith("Firmware version:"):
            firmware_version = line.split(":", 1)[1].strip() or None
        elif line.startswith("Firmware description:"):
            firmware_description = line.split(":", 1)[1].strip() or None
    return {
        "device_id": device_id,
        "firmware_version": firmware_version,
        "firmware_description": firmware_description,
    }


def parse_settings_txt(path: Path, fs: FileSystem = _DEFAULT_FS) -> dict[str, Any]:
    lines = fs.read_text(path).splitlines()
    settings: dict[str, Any] = {}
    recording_periods: list[dict[str, str]] = []
    inside_recording_periods = False
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("recordingPeriods"):
            inside_recording_periods = True
            continue
        if inside_recording_periods:
            if line.startswith("]"):
                inside_recording_periods = False
                settings["recording_periods"] = recording_periods
                continue
            start_match = re.search(r'startTime:\s*"(?P<start>[^"]+)"', line)
            end_match = re.search(r'endTime:\s*"(?P<end>[^"]+)"', line)
            if start_match and end_match:
                recording_periods.append(
                    {
                        "start_time": start_match["start"],
                        "end_time": end_match["end"],
                    }
                )
            continue
        match = _SETTING_LINE_RE.match(line.rstrip(","))
        if match is None:
            continue
        settings[match["key"]] = _parse_scalar_setting(match["value"])
    if recording_periods and "recording_periods" not in settings:
        settings["recording_periods"] = recording_periods
    return settings


def recording_track_point_at_offset(
    recording: SourceRecording,
    offset_seconds: float,
) -> RecorderTrackPoint | None:
    if not recording.track_points:
        return None
    offset_index = max(0, min(int(round(offset_seconds)), len(recording.track_points) - 1))
    return recording.track_points[offset_index]


def recording_path(
    collection_root: Path,
    path: Path,
) -> str:
    root_posix = PurePosixPath(collection_root.as_posix())
    path_posix = PurePosixPath(path.as_posix())
    try:
        relative = path_posix.relative_to(root_posix)
    except ValueError as exc:
        raise ValueError(f"Path {path} is not under {collection_root}") from exc
    return str(relative)


def load_recordings_payload(path: Path, fs: FileSystem = _DEFAULT_FS) -> CollectionRecordings:
    payload = read_json(path, fs)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return CollectionRecordings.model_validate(payload)


def _decimal_coordinate(
    deg_raw: str | None,
    min_raw: str | None,
    dir_raw: str | None,
) -> float | None:
    if not deg_raw or not min_raw or not dir_raw:
        return None
    degrees = float(deg_raw)
    minutes = float(min_raw)
    direction = dir_raw.strip().upper()
    value = degrees + minutes / 60.0
    if direction in {"S", "W"}:
        value *= -1.0
    return value


def _parse_scalar_setting(raw_value: str) -> Any:
    value = raw_value.strip().rstrip(",")
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"0", "1"}:
        return value == "1"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _int_value(raw_value: str | None) -> int:
    if raw_value is None or raw_value == "":
        return 0
    return int(raw_value)


def _optional_text(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _required_text(raw_value: str | None, field_name: str) -> str:
    value = _optional_text(raw_value)
    if value is None:
        raise ValueError(f"Missing required value for {field_name}")
    return value

from __future__ import annotations

import wave
from pathlib import Path

from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.raw_import import find_raw_batch_dirs, import_raw_batches
from alpaca_pipelines.recordings import (
    load_collection_recordings,
    parse_audiomoth_csv,
    parse_device_txt,
    parse_settings_txt,
)


def test_parse_audiomoth_csv_normalizes_gps_and_counters(tmp_path: Path) -> None:
    csv_path = tmp_path / "20250211_075558.CSV"
    csv_path.write_text(
        "\n".join(
            [
                "PPS_NUMBER,AUDIOMOTH_TIME,SAMPLES,TOTAL_SAMPLES,TIMER_COUNT,BUFFERS_FILLED,BUFFERS_WRITTEN,LAST_RMC_AUDIOMOTH_TIME,LAST_RMC_GPS_TIME,STATUS,LAT_DEG,LAT_MIN,LAT_DIR,LONG_DEG,LONG_MIN,LONG_DIR",
                "1,2025-02-11T07:55:58.000,48000,48000,1,4,4,2025-02-11T07:55:58.000,2025-02-11T07:55:58.000,A,51,30.0,N,9,45.0,E",
                "2,2025-02-11T07:55:59.000,47999,95999,2,8,8,,,,,,,",
            ]
        ),
        encoding="utf-8",
    )

    points = parse_audiomoth_csv(csv_path)

    assert [point.total_samples for point in points] == [48000, 95999]
    assert points[0].gps_status == "A"
    assert points[0].latitude == 51.5
    assert points[0].longitude == 9.75
    assert points[1].gps_status is None


def test_parse_device_and_settings_txt_extract_fields(tmp_path: Path) -> None:
    device_path = tmp_path / "DEVICE.TXT"
    settings_path = tmp_path / "SETTINGS.txt"
    device_path.write_text(
        "\n".join(
            [
                "Device: AudioMoth 1A2B3C",
                "Firmware version: 1.11.0",
                "Firmware description: GPS Sync",
            ]
        ),
        encoding="utf-8",
    )
    settings_path.write_text(
        "\n".join(
            [
                "sampleRate: 48000,",
                "gain: 2,",
                "enableGPS: 1,",
                "recordingPeriods: [",
                '  { startTime: "07:30", endTime: "10:30" },',
                "]",
            ]
        ),
        encoding="utf-8",
    )

    device = parse_device_txt(device_path)
    settings = parse_settings_txt(settings_path)

    assert device == {
        "device_id": "AudioMoth 1A2B3C",
        "firmware_version": "1.11.0",
        "firmware_description": "GPS Sync",
    }
    assert settings["sampleRate"] == 48000
    assert settings["gain"] == 2
    assert settings["enableGPS"] is True
    assert settings["recording_periods"] == [{"start_time": "07:30", "end_time": "10:30"}]


def test_import_raw_batches_copies_sidecars_and_recordings(tmp_path: Path) -> None:
    root = tmp_path / "collection-root"
    raw_batch_dir = root / "401_m28_20250213"
    legacy_collection_dir = root / "audio_collection_legacy"
    root.mkdir()
    raw_batch_dir.mkdir()
    legacy_collection_dir.mkdir()

    _write_wav(raw_batch_dir / "20250211_075558.WAV", sample_rate=48000, duration_seconds=1)
    (raw_batch_dir / "20250211_075558.CSV").write_text(
        "\n".join(
            [
                "PPS_NUMBER,AUDIOMOTH_TIME,SAMPLES,TOTAL_SAMPLES,TIMER_COUNT,BUFFERS_FILLED,BUFFERS_WRITTEN,LAST_RMC_AUDIOMOTH_TIME,LAST_RMC_GPS_TIME,STATUS,LAT_DEG,LAT_MIN,LAT_DIR,LONG_DEG,LONG_MIN,LONG_DIR",
                "1,2025-02-11T07:55:58.000,48000,48000,1,4,4,2025-02-11T07:55:58.000,2025-02-11T07:55:58.000,A,51,30.0,N,9,45.0,E",
            ]
        ),
        encoding="utf-8",
    )
    (raw_batch_dir / "DEVICE.TXT").write_text(
        "Device: AudioMoth 1A2B3C\nFirmware version: 1.11.0\nFirmware description: GPS Sync\n",
        encoding="utf-8",
    )
    (raw_batch_dir / "SETTINGS.txt").write_text("sampleRate: 48000,\n", encoding="utf-8")
    (raw_batch_dir / "LOG.TXT").write_text("log", encoding="utf-8")

    identity_map = IdentityMap.model_validate(
        {
            "canonical": {"401": {"display_name": "401"}},
            "aliases": {"401": "401"},
        }
    )

    raw_batches = find_raw_batch_dirs(root)
    result = import_raw_batches(root, identity_map)
    imported_collection_dir = root / "audio_collection_401_m28_20250213"
    recordings = load_collection_recordings(imported_collection_dir)

    assert [path.name for path in raw_batches] == ["401_m28_20250213"]
    assert result.imported_batches == ["401_m28_20250213"]
    assert result.matched_csv_count == 1
    assert result.missing_csv_count == 0
    assert len(recordings) == 1
    assert recordings[0].key == "401_20250211_075558"
    assert recordings[0].deployment_token == "m28"
    assert recordings[0].csv_path == (
        "audio_collection_401_m28_20250213/raw_recordings/20250211_075558.CSV"
    )
    assert recordings[0].device_id == "AudioMoth 1A2B3C"
    assert recordings[0].sample_rate == 48000
    assert recordings[0].track_points is not None
    assert recordings[0].track_points[0].latitude == 51.5
    assert (imported_collection_dir / "raw_recordings" / "LOG.TXT").is_file()


def test_import_raw_batches_accepts_canonical_subject_without_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collection-root"
    raw_batch_dir = root / "408_m25_20250213"
    root.mkdir()
    raw_batch_dir.mkdir()

    _write_wav(raw_batch_dir / "20250211_075558.WAV", sample_rate=48000, duration_seconds=1)

    identity_map = IdentityMap.model_validate(
        {
            "canonical": {"408": {"display_name": "408"}},
            "aliases": {},
        }
    )

    result = import_raw_batches(root, identity_map)
    recordings = load_collection_recordings(root / "audio_collection_408_m25_20250213")

    assert result.imported_batches == ["408_m25_20250213"]
    assert len(recordings) == 1
    assert recordings[0].subject_id == "408"
    assert recordings[0].track_points is None


def _write_wav(path: Path, sample_rate: int, duration_seconds: int) -> None:
    frame_count = sample_rate * duration_seconds
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)

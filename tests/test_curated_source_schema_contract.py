from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from alpaca_pipelines.prediction.review import CuratedPredictionSourceManifest


def test_curated_source_schema_is_committed_and_loadable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "contracts" / "json-schema" / "CuratedPredictionSourceManifest.json"

    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema


def test_malformed_curated_source_manifest_fails_validation_with_clear_field_name() -> None:
    malformed_payload = {
        "schema_version": 1,
        "source_type": "manual_review_curated",
        "collection_name": "audio_collection_alpha",
        "source_category_dir": "raw_recordings",
        "source_display_path": "audio_collection_alpha/raw_recordings/audio.wav",
        "source_recording_key": "401_20250211_075558",
        "source_audio_file": "/tmp/audio.wav",
        "prediction_run_id": "run-1",
        "review_session_id": "session-1",
        "created_at": "2026-04-21T09:00:00Z",
        "items": [
            {
                "curated_example_id": "example-1",
                "review_item_id": "item-1",
                "start_s": 0.1,
                "end_s": 0.7,
                "duration_s": 0.6,
                "label": "target",
                "snippet_wav_path": "/tmp/snippet.wav",
                "source_recording_key": "401_20250211_075558",
                "source_collection_name": "audio_collection_alpha",
                "source_category_dir": "raw_recordings",
                "source_display_path": "audio_collection_alpha/raw_recordings/audio.wav",
                "source_audio_file": "/tmp/audio.wav",
                "prediction_run_id": "run-1",
                "review_session_id": "session-1",
            }
        ],
    }

    with pytest.raises(ValidationError, match="source_relative_path"):
        CuratedPredictionSourceManifest.model_validate(malformed_payload)

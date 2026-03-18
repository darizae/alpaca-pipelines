from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from alpaca_pipelines.recordings import SourceRecording

VALID_REVIEW_ANNOTATIONS: frozenset[str] = frozenset({"target", "noise", "discard"})

Classification = Literal["target", "noise"]
SourceType = Literal["hum", "mined_source", "low_quality_hum"]
ReviewStatus = Literal["pending", "approved", "rejected"]
SplitName = Literal["train", "val", "test"]


class SnippetEntry(BaseModel):
    uid: int
    filename: str
    classification: Classification
    source_type: SourceType
    source_path: str
    start_s: float
    end_s: float
    duration_s: float
    quality: int | None
    subject_id: str | None
    recording_date: str | None
    collection: str
    session_key: str | None
    recording_time: str | None = None
    source_recording_key: str | None = None
    source_recording_start_s: float | None = None
    source_recording_end_s: float | None = None
    snippet_started_at: str | None = None
    snippet_ended_at: str | None = None
    snippet_midpoint_latitude: float | None = None
    snippet_midpoint_longitude: float | None = None
    snippet_gps_status: str | None = None
    split: SplitName | None = None
    review_status: ReviewStatus = "pending"


class ManifestMeta(BaseModel):
    strategy_name: str
    created_at: str
    collection_root: str
    merged_index_path: str
    seed: int
    n_snippets: int
    n_target: int
    n_noise: int
    n_recordings: int = 0
    n_recordings_with_sidecar: int = 0
    manifest_hash: str = ""
    strategy_config: dict[str, object] | None = None


class Manifest(BaseModel):
    meta: ManifestMeta
    snippets: list[SnippetEntry]
    recordings: list[SourceRecording] = Field(default_factory=list)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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
    manifest_hash: str = ""
    strategy_config: dict[str, object] | None = None


class Manifest(BaseModel):
    meta: ManifestMeta
    snippets: list[SnippetEntry]

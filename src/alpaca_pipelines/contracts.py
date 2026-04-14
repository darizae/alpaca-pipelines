"""
Data contracts for the persistence layer and pipeline state.

These models define the exact JSON shapes that alpaca-pipelines reads
from the HPC persistence layer and writes for its own run state.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from alpaca_pipelines.recordings import SourceRecording

# ---------------------------------------------------------------------------
# Persistence layer contracts (read-only inputs)
# ---------------------------------------------------------------------------


class IndexMeta(BaseModel):
    """Metadata block of merged_index.json."""

    generated_at: str | None = None
    n_collections: int
    n_total_hums: int
    n_recordings: int = 0
    n_recordings_with_sidecar: int = 0


class IndexEntry(BaseModel):
    """Single entry in merged_index.json."""

    collection: str
    subject_id: str
    recording_date: str
    recording_time: str | None
    hum_path: str
    hum_start_s: float
    hum_end_s: float
    source_quality: int
    keep: bool
    hum_uid: int
    source_recording_key: str | None = None


class MergedIndex(BaseModel):
    """Top-level merged_index.json structure."""

    meta: IndexMeta
    entries: list[IndexEntry]
    recordings: list[SourceRecording] = Field(default_factory=list)


Classification = Literal["target", "noise"]
SourceType = Literal["hum", "mined_source", "low_quality_hum"]
ReviewStatus = Literal["pending", "approved", "rejected"]
SplitName = Literal["train", "val", "test"]


class ManifestSnippet(BaseModel):
    """Single snippet entry in manifest.json."""

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
    """Metadata block of manifest.json."""

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
    strategy_config: dict[str, Any] | None = None


class DatasetManifest(BaseModel):
    """Top-level manifest.json structure."""

    meta: ManifestMeta
    snippets: list[ManifestSnippet]
    recordings: list[SourceRecording] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run state contracts (owned by alpaca-pipelines)
# ---------------------------------------------------------------------------

RunType = Literal["training", "prediction", "evaluation", "rf_training"]
RunStatus = Literal["created", "submitted", "running", "completed", "failed", "cancelled"]
PredictionProgressStage = Literal[
    "initializing",
    "resolving_inputs",
    "predicting",
    "rf_filtering",
    "writing_summary",
    "completed",
]
WorkflowName = Literal["standardizer", "dataset_builder"]
WorkflowOperationKind = Literal[
    "import",
    "scan",
    "plan",
    "apply",
    "index",
    "build",
    "prepare_review",
    "apply_review",
]
WorkflowOperationStatus = Literal["pending", "running", "completed", "failed"]


class RunOutputs(BaseModel):
    """Pointers to output artifacts produced by a run.

    Directory pointers (``*_dir``) are set at run creation time.
    File pointers are set by executors when the artifact is actually produced.
    """

    trained_model_path: str | None = None
    rf_model_path: str | None = None
    model_dir: str | None = None
    predictions_dir: str | None = None
    prediction_selection_tables_dir: str | None = None
    prediction_selection_tables_summary_path: str | None = None
    evaluation_dir: str | None = None
    summaries_dir: str | None = None
    tensorboard_dir: str | None = None
    log_dir: str | None = None
    rf_filtered: bool = False


class PredictionProgress(BaseModel):
    """Structured progress snapshot for prediction runs."""

    stage: PredictionProgressStage
    files_total: int | None = None
    files_completed: int | None = None
    current_file: str | None = None
    current_file_windows_total: int | None = None
    current_file_windows_completed: int | None = None
    detections_so_far: int | None = None
    updated_at: str | None = None


class RunProgress(BaseModel):
    """Optional progress tracking for long-running jobs."""

    current_epoch: int | None = None
    total_epochs: int | None = None
    current_phase: str | None = None
    best_metric_value: float | None = None
    best_metric_name: str | None = None
    prediction: PredictionProgress | None = None


class RunState(BaseModel):
    """Persistent state of a pipeline run, stored as run_state.json."""

    run_id: str
    run_type: RunType
    status: RunStatus = "created"
    created_at: str
    submitted_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    outputs: RunOutputs = Field(default_factory=RunOutputs)
    progress: RunProgress = Field(default_factory=RunProgress)
    error_message: str | None = None
    slurm_job_id: str | None = None
    run_dir: str = ""


class WorkflowOperation(BaseModel):
    """Persistent state of a non-run workflow operation."""

    job_id: str
    workflow: WorkflowName
    kind: WorkflowOperationKind
    status: WorkflowOperationStatus = "pending"
    created_at: str
    started_at: str
    finished_at: str | None = None
    job_dir: str
    artifact_path: str | None = None
    rollback_artifact_path: str | None = None
    result_summary: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Directory name constants
# ---------------------------------------------------------------------------

SNIPPETS_DIR: str = "snippets"
SPLITS_DIR: str = "splits"
MANIFEST_FILENAME: str = "manifest.json"
RUN_STATE_FILENAME: str = "run_state.json"
LOGS_DIR: str = "logs"
OUTPUTS_DIR: str = "outputs"
SLURM_DIR: str = "slurm"
MODEL_DIR: str = "model"
PREDICTIONS_DIR: str = "predictions"
PREDICTION_SELECTION_TABLES_DIR: str = "selection_tables"
PREDICTION_SELECTION_TABLES_SUMMARY_FILENAME: str = "selection_tables_summary.json"
EVALUATION_DIR: str = "evaluation"
SUMMARIES_DIR: str = "summaries"
TRAINING_SUMMARY_FILENAME: str = "training_summary.json"
TRAINING_HISTORY_FILENAME: str = "training_history.json"

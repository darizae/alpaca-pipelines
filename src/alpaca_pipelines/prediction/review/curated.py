"""Durable curated prediction-review source materialization."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpaca_pipelines.datasets.audio_utils import extract_segment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review.config import (
    PredictionReviewSessionItem,
    PredictionReviewSessionManifest,
)
from alpaca_pipelines.recordings import derive_source_recording_key_from_stem
from alpaca_pipelines.runs.manager import RunManager

CURATED_SOURCE_TYPE: Literal["manual_review_curated"] = "manual_review_curated"
CURATED_ROOT_DIRNAME = "_curated_prediction_examples"
CURATED_MANIFEST_FILENAME = "manifest.json"
CURATED_SNIPPETS_DIRNAME = "snippets"


class CuratedLabelAssignments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    labels: dict[str, Literal["target", "noise"]]

    @model_validator(mode="after")
    def validate_labels(self) -> "CuratedLabelAssignments":
        if not self.labels:
            raise ValueError("labels must contain at least one entry")
        return self


class CuratedPredictionSourceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curated_example_id: str
    review_item_id: str
    detection_index: int | None = None
    start_s: float
    end_s: float
    duration_s: float
    detection_score: float | None = None
    label: Literal["target", "noise"]
    snippet_wav_path: str
    source_recording_key: str
    source_collection_name: str
    source_category_dir: str
    source_relative_path: str
    source_audio_file: str
    prediction_run_id: str
    review_session_id: str
    provenance_type: Literal["manual_review_curated"] = CURATED_SOURCE_TYPE
    payload_json: dict[str, object] | None = None


class CuratedPredictionSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_type: Literal["manual_review_curated"] = CURATED_SOURCE_TYPE
    collection_name: str
    source_category_dir: str | None = None
    source_relative_path: str | None = None
    source_recording_key: str | None = None
    source_audio_file: str | None = None
    prediction_run_id: str
    review_session_id: str
    created_at: str
    items: list[CuratedPredictionSourceItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "CuratedPredictionSourceManifest":
        if not self.items:
            raise ValueError("items must contain at least one curated entry")
        return self


def curated_sources_root(
    *,
    datasets_root: Path,
    destination_root: Path | None = None,
) -> Path:
    return (
        destination_root if destination_root is not None else datasets_root / CURATED_ROOT_DIRNAME
    )


def materialize_curated_prediction_examples(
    *,
    run_manager: RunManager,
    collection_root: Path,
    datasets_root: Path,
    manifest_path: Path,
    labels_path: Path,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    review_manifest = _load_review_manifest(manifest_path)
    _validate_prediction_run(run_manager, review_manifest.prediction_run_id)
    labels = _load_labels(labels_path)

    labeled_items: list[tuple[PredictionReviewSessionItem, Literal["target", "noise"]]] = []
    for item in review_manifest.items:
        label = labels.get(item.item_id)
        if label in {"target", "noise"}:
            labeled_items.append((item, label))
    if not labeled_items:
        raise ValueError("No eligible labeled review items to materialize")

    root = curated_sources_root(datasets_root=datasets_root, destination_root=destination_root)
    root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[tuple[PredictionReviewSessionItem, Literal["target", "noise"]]]] = (
        defaultdict(list)
    )
    for item, label in labeled_items:
        source_collection_name, _, _, _ = _resolve_source_fields(item, collection_root)
        groups[source_collection_name].append((item, label))

    manifests_created: list[str] = []
    by_label: dict[str, int] = {"target": 0, "noise": 0}
    created_count = 0
    updated_count = 0
    skipped_count = 0
    source_recording_keys: set[str] = set()

    for collection_name, grouped_items in groups.items():
        session_dir = (
            root / collection_name / review_manifest.prediction_run_id / review_manifest.session_id
        )
        snippets_dir = session_dir / CURATED_SNIPPETS_DIRNAME
        snippets_dir.mkdir(parents=True, exist_ok=True)
        manifest_out_path = session_dir / CURATED_MANIFEST_FILENAME
        existing_items_by_id = _load_existing_items(manifest_out_path)

        new_items: list[CuratedPredictionSourceItem] = []
        for item, label in sorted(grouped_items, key=lambda pair: pair[0].item_id):
            (
                source_collection_name,
                source_category_dir,
                source_relative_path,
                source_recording_key,
            ) = _resolve_source_fields(item, collection_root)
            source_recording_keys.add(source_recording_key)

            curated_example_id = _build_curated_example_id(
                prediction_run_id=review_manifest.prediction_run_id,
                review_session_id=review_manifest.session_id,
                item_id=item.item_id,
            )
            snippet_filename = f"{label}_{curated_example_id}.wav"
            snippet_path = snippets_dir / snippet_filename

            previous = existing_items_by_id.get(curated_example_id)
            needs_update = (
                previous is None
                or previous.label != label
                or previous.start_s != item.start_s
                or previous.end_s != item.end_s
                or previous.source_recording_key != source_recording_key
                or previous.source_audio_file != item.audio_file
                or not snippet_path.is_file()
            )
            if needs_update:
                duration_s = extract_segment(
                    source_path=Path(item.audio_file),
                    start_s=item.start_s,
                    end_s=item.end_s,
                    destination_path=snippet_path,
                )
                if previous is None:
                    created_count += 1
                else:
                    updated_count += 1
            else:
                if previous is None:
                    raise ValueError(
                        "Internal error: expected existing curated item for {}".format(
                            curated_example_id
                        )
                    )
                duration_s = previous.duration_s
                skipped_count += 1

            by_label[label] += 1
            new_items.append(
                CuratedPredictionSourceItem(
                    curated_example_id=curated_example_id,
                    review_item_id=item.review_item_id or item.item_id,
                    detection_index=item.detection_index,
                    start_s=item.start_s,
                    end_s=item.end_s,
                    duration_s=duration_s,
                    detection_score=item.detection_score,
                    label=label,
                    snippet_wav_path=str(snippet_path),
                    source_recording_key=source_recording_key,
                    source_collection_name=source_collection_name,
                    source_category_dir=source_category_dir,
                    source_relative_path=source_relative_path,
                    source_audio_file=item.audio_file,
                    prediction_run_id=review_manifest.prediction_run_id,
                    review_session_id=review_manifest.session_id,
                    payload_json=item.payload_json,
                )
            )

        template_item = new_items[0]
        manifest = CuratedPredictionSourceManifest(
            collection_name=collection_name,
            source_category_dir=template_item.source_category_dir,
            source_relative_path=template_item.source_relative_path,
            source_recording_key=template_item.source_recording_key,
            source_audio_file=template_item.source_audio_file,
            prediction_run_id=review_manifest.prediction_run_id,
            review_session_id=review_manifest.session_id,
            created_at=_now_iso(),
            items=new_items,
        )
        write_json(manifest_out_path, manifest.model_dump(mode="json"))
        manifests_created.append(str(manifest_out_path))

    return {
        "curated_source_root": str(root),
        "manifest_paths": manifests_created,
        "prediction_run_id": review_manifest.prediction_run_id,
        "review_session_id": review_manifest.session_id,
        "counts_by_label": by_label,
        "created_count": created_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "total_items": len(labeled_items),
        "source_recording_keys": sorted(source_recording_keys),
    }


def list_curated_prediction_sources(
    *,
    datasets_root: Path,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    root = curated_sources_root(datasets_root=datasets_root, destination_root=destination_root)
    if not root.exists():
        return {
            "curated_source_root": str(root),
            "manifests": [],
            "counts_by_collection": {},
            "counts_by_label": {},
            "counts_by_provenance_type": {},
            "warnings": [],
        }

    manifests: list[dict[str, Any]] = []
    counts_by_collection: dict[str, int] = defaultdict(int)
    counts_by_label: dict[str, int] = defaultdict(int)
    counts_by_provenance_type: dict[str, int] = defaultdict(int)
    warnings: list[str] = []

    for manifest_path in sorted(root.rglob(CURATED_MANIFEST_FILENAME)):
        try:
            raw_payload = read_json(manifest_path)
            if not isinstance(raw_payload, dict):
                raise ValueError("Expected JSON object")
            manifest = CuratedPredictionSourceManifest.model_validate(raw_payload)
        except Exception as exc:
            warnings.append(f"Invalid manifest {manifest_path}: {exc}")
            continue

        counts_by_collection[manifest.collection_name] += len(manifest.items)
        counts_by_provenance_type[manifest.source_type] += len(manifest.items)
        for item in manifest.items:
            counts_by_label[item.label] += 1
            if not Path(item.snippet_wav_path).is_file():
                warnings.append(
                    "Missing snippet wav for curated example {}: {}".format(
                        item.curated_example_id, item.snippet_wav_path
                    )
                )

        manifests.append(
            {
                "manifest_path": str(manifest_path),
                "collection_name": manifest.collection_name,
                "prediction_run_id": manifest.prediction_run_id,
                "review_session_id": manifest.review_session_id,
                "n_items": len(manifest.items),
                "counts_by_label": {
                    "target": sum(1 for item in manifest.items if item.label == "target"),
                    "noise": sum(1 for item in manifest.items if item.label == "noise"),
                },
            }
        )

    return {
        "curated_source_root": str(root),
        "manifests": manifests,
        "counts_by_collection": dict(sorted(counts_by_collection.items())),
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_provenance_type": dict(sorted(counts_by_provenance_type.items())),
        "warnings": warnings,
    }


def _load_review_manifest(manifest_path: Path) -> PredictionReviewSessionManifest:
    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest file not found: {}".format(manifest_path))
    raw_payload = read_json(manifest_path)
    if not isinstance(raw_payload, dict):
        raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
    return PredictionReviewSessionManifest.model_validate(raw_payload)


def _validate_prediction_run(run_manager: RunManager, prediction_run_id: str) -> None:
    run_state = run_manager.find_run(prediction_run_id)
    if run_state.run_type != "prediction":
        raise ValueError("Expected prediction run, got: {}".format(run_state.run_type))
    if run_state.status != "completed":
        raise ValueError(
            "Prediction run must be completed for curated materialization, status: {}".format(
                run_state.status
            )
        )


def _load_labels(labels_path: Path) -> dict[str, Literal["target", "noise"]]:
    if not labels_path.is_file():
        raise FileNotFoundError("Curated labels file not found: {}".format(labels_path))
    payload = read_json(labels_path)
    if isinstance(payload, dict) and "labels" in payload:
        contract = CuratedLabelAssignments.model_validate(payload)
        return contract.labels
    if isinstance(payload, dict):
        validated: dict[str, Literal["target", "noise"]] = {}
        for key, value in payload.items():
            if value in {"target", "noise"}:
                validated[str(key)] = value
        if validated:
            return validated
    raise ValueError(
        "Invalid curated labels payload. Expected {'labels': {'item_id': 'target|noise'}}"
    )


def _load_existing_items(manifest_path: Path) -> dict[str, CuratedPredictionSourceItem]:
    if not manifest_path.is_file():
        return {}
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
    manifest = CuratedPredictionSourceManifest.model_validate(payload)
    return {item.curated_example_id: item for item in manifest.items}


def _resolve_source_fields(
    item: PredictionReviewSessionItem,
    collection_root: Path,
) -> tuple[str, str, str, str]:
    source_path = Path(item.audio_file)

    collection_name = item.source_collection_name
    source_category_dir = item.source_category_dir
    source_relative_path = item.source_relative_path
    source_recording_key = item.source_recording_key

    if collection_name is None or source_category_dir is None or source_relative_path is None:
        resolved_source = source_path.resolve()
        resolved_root = collection_root.resolve()
        try:
            rel = resolved_source.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "Missing source collection metadata for audio outside collection root: "
                f"{source_path}"
            ) from exc
        parts = rel.parts
        if len(parts) < 3:
            raise ValueError(
                "Could not infer source metadata from path under collection root: {}".format(
                    source_path
                )
            )
        if collection_name is None:
            collection_name = parts[0]
        if source_category_dir is None:
            source_category_dir = parts[1]
        if source_relative_path is None:
            source_relative_path = str(rel).replace("\\", "/")

    if source_recording_key is None:
        source_recording_key = _derive_source_recording_key(
            collection_name=collection_name,
            source_path=source_path,
        )
    if source_recording_key is None:
        raise ValueError(
            "Missing source_recording_key for review item {}; provide source_recording_key in "
            "review manifest item".format(item.item_id)
        )

    return collection_name, source_category_dir, source_relative_path, source_recording_key


def _derive_source_recording_key(collection_name: str, source_path: Path) -> str | None:
    if not collection_name.startswith("audio_collection_"):
        return None
    stem = source_path.stem
    suffix = collection_name[len("audio_collection_") :]
    if not suffix:
        return None
    subject_id = suffix.split("_", 1)[0]
    if not subject_id:
        return None
    try:
        return derive_source_recording_key_from_stem(subject_id, stem)
    except ValueError:
        return None


def _build_curated_example_id(
    *,
    prediction_run_id: str,
    review_session_id: str,
    item_id: str,
) -> str:
    payload = f"{prediction_run_id}|{review_session_id}|{item_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

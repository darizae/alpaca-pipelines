"""Durable curated prediction-review source materialization."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpaca_pipelines.datasets.audio_utils import extract_segment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review.config import (
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


class CuratedPredictionExportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curated_example_id: str | None = None
    review_item_id: str
    source_audio_file: str
    start_s: float
    end_s: float
    label: Literal["target", "noise"]
    detection_index: int | None = None
    detection_score: float | None = None
    source_collection_name: str | None = None
    source_category_dir: str | None = None
    source_relative_path: str | None = None
    source_recording_key: str | None = None
    payload_json: dict[str, object] | None = None


class CuratedPredictionExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    prediction_run_id: str
    review_session_id: str
    source_collection_name: str | None = None
    source_category_dir: str | None = None
    source_relative_path: str | None = None
    source_recording_key: str | None = None
    source_audio_file: str | None = None
    items: list[CuratedPredictionExportItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "CuratedPredictionExportManifest":
        if not self.items:
            raise ValueError("items must contain at least one curated export item")
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
    source_display_path: str
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
    source_category_dir: str
    source_relative_path: str
    source_display_path: str
    source_recording_key: str
    source_audio_file: str
    prediction_run_id: str
    review_session_id: str
    created_at: str
    items: list[CuratedPredictionSourceItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "CuratedPredictionSourceManifest":
        if not self.items:
            raise ValueError("items must contain at least one curated entry")
        return self


class _MaterializationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curated_example_id: str | None = None
    review_item_id: str
    source_audio_file: str
    start_s: float
    end_s: float
    label: Literal["target", "noise"]
    detection_index: int | None = None
    detection_score: float | None = None
    source_collection_name: str
    source_category_dir: str
    source_relative_path: str
    source_display_path: str
    source_recording_key: str
    payload_json: dict[str, object] | None = None


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
    manifest_path: Path | None = None,
    labels_path: Path | None = None,
    curated_export_manifest: Path | None = None,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    prediction_run_id: str
    review_session_id: str
    materialization_items: list[_MaterializationItem]

    if curated_export_manifest is not None:
        if manifest_path is not None or labels_path is not None:
            raise ValueError(
                "curated_export_manifest mode does not allow manifest_path or labels_path"
            )
        curated_export = _load_curated_export_manifest(curated_export_manifest)
        _validate_prediction_run(run_manager, curated_export.prediction_run_id)
        prediction_run_id = curated_export.prediction_run_id
        review_session_id = curated_export.review_session_id
        materialization_items = _materialization_items_from_curated_export(
            curated_export=curated_export,
            collection_root=collection_root,
        )
    else:
        if manifest_path is None or labels_path is None:
            raise ValueError("Provide either manifest_path+labels_path or curated_export_manifest")
        review_manifest = _load_review_manifest(manifest_path)
        _validate_prediction_run(run_manager, review_manifest.prediction_run_id)
        labels = _load_labels(labels_path)
        prediction_run_id = review_manifest.prediction_run_id
        review_session_id = review_manifest.session_id
        materialization_items = _materialization_items_from_review_manifest(
            review_manifest=review_manifest,
            labels=labels,
            collection_root=collection_root,
        )

    if not materialization_items:
        raise ValueError("No eligible labeled review items to materialize")

    root = curated_sources_root(datasets_root=datasets_root, destination_root=destination_root)
    root.mkdir(parents=True, exist_ok=True)

    groups: dict[
        tuple[str, str, str, str, str, str],
        list[_MaterializationItem],
    ] = defaultdict(list)
    for item in materialization_items:
        key = (
            item.source_collection_name,
            item.source_category_dir,
            item.source_relative_path,
            item.source_display_path,
            item.source_recording_key,
            item.source_audio_file,
        )
        groups[key].append(item)

    manifests_created: list[str] = []
    by_label: dict[str, int] = {"target": 0, "noise": 0}
    created_count = 0
    updated_count = 0
    skipped_count = 0
    source_recording_keys: set[str] = set()

    for (
        collection_name,
        source_category_dir,
        source_relative_path,
        source_display_path,
        source_recording_key,
        source_audio_file,
    ), grouped_items in sorted(groups.items()):
        session_dir = (
            root / collection_name / prediction_run_id / review_session_id / source_recording_key
        )
        snippets_dir = session_dir / CURATED_SNIPPETS_DIRNAME
        snippets_dir.mkdir(parents=True, exist_ok=True)
        manifest_out_path = session_dir / CURATED_MANIFEST_FILENAME
        existing_items_by_id = _load_existing_items(manifest_out_path)

        new_items: list[CuratedPredictionSourceItem] = []
        for item in sorted(grouped_items, key=lambda value: value.review_item_id):
            source_recording_keys.add(item.source_recording_key)

            if item.curated_example_id:
                curated_example_id = item.curated_example_id
            else:
                curated_example_id = _build_curated_example_id(
                    prediction_run_id=prediction_run_id,
                    review_session_id=review_session_id,
                    review_item_id=item.review_item_id,
                )
            snippet_filename = f"{item.label}_{curated_example_id}.wav"
            snippet_path = snippets_dir / snippet_filename

            previous = existing_items_by_id.get(curated_example_id)
            needs_update = (
                previous is None
                or previous.label != item.label
                or previous.start_s != item.start_s
                or previous.end_s != item.end_s
                or previous.source_recording_key != item.source_recording_key
                or previous.source_audio_file != item.source_audio_file
                or not snippet_path.is_file()
            )
            if needs_update:
                duration_s = extract_segment(
                    source_path=Path(item.source_audio_file),
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

            by_label[item.label] += 1
            new_items.append(
                CuratedPredictionSourceItem(
                    curated_example_id=curated_example_id,
                    review_item_id=item.review_item_id,
                    detection_index=item.detection_index,
                    start_s=item.start_s,
                    end_s=item.end_s,
                    duration_s=duration_s,
                    detection_score=item.detection_score,
                    label=item.label,
                    snippet_wav_path=str(snippet_path),
                    source_recording_key=item.source_recording_key,
                    source_collection_name=item.source_collection_name,
                    source_category_dir=item.source_category_dir,
                    source_relative_path=item.source_relative_path,
                    source_display_path=item.source_display_path,
                    source_audio_file=item.source_audio_file,
                    prediction_run_id=prediction_run_id,
                    review_session_id=review_session_id,
                    payload_json=item.payload_json,
                )
            )

        manifest = CuratedPredictionSourceManifest(
            collection_name=collection_name,
            source_category_dir=source_category_dir,
            source_relative_path=source_relative_path,
            source_display_path=source_display_path,
            source_recording_key=source_recording_key,
            source_audio_file=source_audio_file,
            prediction_run_id=prediction_run_id,
            review_session_id=review_session_id,
            created_at=_now_iso(),
            items=new_items,
        )
        write_json(manifest_out_path, manifest.model_dump(mode="json"))
        manifests_created.append(str(manifest_out_path))

    return {
        "curated_source_root": str(root),
        "manifest_paths": manifests_created,
        "prediction_run_id": prediction_run_id,
        "review_session_id": review_session_id,
        "counts_by_label": by_label,
        "created_count": created_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "total_items": len(materialization_items),
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


def _load_curated_export_manifest(manifest_path: Path) -> CuratedPredictionExportManifest:
    if not manifest_path.is_file():
        raise FileNotFoundError("Curated export manifest file not found: {}".format(manifest_path))
    raw_payload = read_json(manifest_path)
    if not isinstance(raw_payload, dict):
        raise ValueError(
            "Expected JSON object in curated export manifest: {}".format(manifest_path)
        )
    return CuratedPredictionExportManifest.model_validate(raw_payload)


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
    if not isinstance(payload, dict):
        raise ValueError(
            "Invalid curated labels payload. Expected {'labels': {'item_id': 'target|noise'}}"
        )
    contract = CuratedLabelAssignments.model_validate(payload)
    return contract.labels


def _load_existing_items(manifest_path: Path) -> dict[str, CuratedPredictionSourceItem]:
    if not manifest_path.is_file():
        return {}
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
    manifest = CuratedPredictionSourceManifest.model_validate(payload)
    return {item.curated_example_id: item for item in manifest.items}


def _materialization_items_from_review_manifest(
    *,
    review_manifest: PredictionReviewSessionManifest,
    labels: dict[str, Literal["target", "noise"]],
    collection_root: Path,
) -> list[_MaterializationItem]:
    materialization_items: list[_MaterializationItem] = []
    for review_item in review_manifest.items:
        label = labels.get(review_item.item_id)
        if label not in {"target", "noise"}:
            continue
        materialization_items.append(
            _build_materialization_item(
                review_item_id=review_item.review_item_id or review_item.item_id,
                curated_example_id=None,
                source_audio_file=review_item.audio_file,
                start_s=review_item.start_s,
                end_s=review_item.end_s,
                label=label,
                detection_index=review_item.detection_index,
                detection_score=review_item.detection_score,
                source_collection_name=review_item.source_collection_name,
                source_category_dir=review_item.source_category_dir,
                source_relative_path=review_item.source_relative_path,
                source_recording_key=review_item.source_recording_key,
                payload_json=review_item.payload_json,
                collection_root=collection_root,
            )
        )
    return materialization_items


def _materialization_items_from_curated_export(
    *,
    curated_export: CuratedPredictionExportManifest,
    collection_root: Path,
) -> list[_MaterializationItem]:
    materialization_items: list[_MaterializationItem] = []
    for item in curated_export.items:
        materialization_items.append(
            _build_materialization_item(
                review_item_id=item.review_item_id,
                curated_example_id=item.curated_example_id,
                source_audio_file=item.source_audio_file,
                start_s=item.start_s,
                end_s=item.end_s,
                label=item.label,
                detection_index=item.detection_index,
                detection_score=item.detection_score,
                source_collection_name=item.source_collection_name
                or curated_export.source_collection_name,
                source_category_dir=item.source_category_dir or curated_export.source_category_dir,
                source_relative_path=item.source_relative_path
                or curated_export.source_relative_path,
                source_recording_key=item.source_recording_key
                or curated_export.source_recording_key,
                payload_json=item.payload_json,
                collection_root=collection_root,
            )
        )
    return materialization_items


def _build_materialization_item(
    *,
    review_item_id: str,
    curated_example_id: str | None,
    source_audio_file: str,
    start_s: float,
    end_s: float,
    label: Literal["target", "noise"],
    detection_index: int | None,
    detection_score: float | None,
    source_collection_name: str | None,
    source_category_dir: str | None,
    source_relative_path: str | None,
    source_recording_key: str | None,
    payload_json: dict[str, object] | None,
    collection_root: Path,
) -> _MaterializationItem:
    (
        resolved_collection_name,
        resolved_category_dir,
        resolved_relative_path,
        resolved_display_path,
        resolved_recording_key,
    ) = _resolve_source_fields(
        source_audio_file=source_audio_file,
        collection_root=collection_root,
        source_collection_name=source_collection_name,
        source_category_dir=source_category_dir,
        source_relative_path=source_relative_path,
        source_recording_key=source_recording_key,
        review_item_id=review_item_id,
    )

    return _MaterializationItem(
        review_item_id=review_item_id,
        curated_example_id=curated_example_id,
        source_audio_file=source_audio_file,
        start_s=start_s,
        end_s=end_s,
        label=label,
        detection_index=detection_index,
        detection_score=detection_score,
        source_collection_name=resolved_collection_name,
        source_category_dir=resolved_category_dir,
        source_relative_path=resolved_relative_path,
        source_display_path=resolved_display_path,
        source_recording_key=resolved_recording_key,
        payload_json=payload_json,
    )


def _resolve_source_fields(
    *,
    source_audio_file: str,
    collection_root: Path,
    source_collection_name: str | None,
    source_category_dir: str | None,
    source_relative_path: str | None,
    source_recording_key: str | None,
    review_item_id: str,
) -> tuple[str, str, str, str, str]:
    source_path = Path(source_audio_file)

    inferred_collection_name: str | None = None
    inferred_category_dir: str | None = None
    inferred_relative_path: str | None = None
    try:
        inferred_collection_name, inferred_category_dir, inferred_relative_path = (
            _infer_source_fields_from_audio_path(
                source_path=source_path,
                collection_root=collection_root,
            )
        )
    except ValueError:
        if (
            source_collection_name is None
            or source_category_dir is None
            or source_relative_path is None
        ):
            raise ValueError(
                "Missing source metadata for review item {} and could not infer from {}".format(
                    review_item_id, source_audio_file
                )
            )

    normalized_collection_name = _normalize_simple_name(
        source_collection_name,
        "source_collection_name",
    )
    normalized_category_dir = _normalize_simple_name(
        source_category_dir,
        "source_category_dir",
    )
    normalized_relative_path = _normalize_relative_path(
        source_relative_path,
        "source_relative_path",
    )

    resolved_collection_name = _resolve_or_inferred(
        supplied=normalized_collection_name,
        inferred=inferred_collection_name,
        field_name="source_collection_name",
        review_item_id=review_item_id,
    )
    resolved_category_dir = _resolve_or_inferred(
        supplied=normalized_category_dir,
        inferred=inferred_category_dir,
        field_name="source_category_dir",
        review_item_id=review_item_id,
    )
    resolved_relative_path = _resolve_or_inferred(
        supplied=normalized_relative_path,
        inferred=inferred_relative_path,
        field_name="source_relative_path",
        review_item_id=review_item_id,
    )

    _validate_relative_path_shape(
        source_relative_path=resolved_relative_path,
        source_collection_name=resolved_collection_name,
        source_category_dir=resolved_category_dir,
        review_item_id=review_item_id,
    )

    if source_recording_key is None:
        source_recording_key = _derive_source_recording_key(
            collection_name=resolved_collection_name,
            source_path=source_path,
        )
    if source_recording_key is None:
        raise ValueError(
            "Missing source_recording_key for review item {}; provide source_recording_key in "
            "input manifest".format(review_item_id)
        )

    source_display_path = (
        f"{resolved_collection_name}/{resolved_category_dir}/{resolved_relative_path}"
    )
    return (
        resolved_collection_name,
        resolved_category_dir,
        resolved_relative_path,
        source_display_path,
        source_recording_key,
    )


def _infer_source_fields_from_audio_path(
    *,
    source_path: Path,
    collection_root: Path,
) -> tuple[str, str, str]:
    resolved_source = source_path.resolve()
    resolved_root = collection_root.resolve()
    rel = resolved_source.relative_to(resolved_root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(
            "Could not infer source metadata from path under collection root: {}".format(
                source_path
            )
        )

    collection_name = parts[0]
    category_dir = parts[1]
    relative_path = str(PurePosixPath(*parts[2:]))
    return collection_name, category_dir, relative_path


def _normalize_simple_name(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty")
    if "/" in stripped or "\\" in stripped:
        raise ValueError(f"{field_name} must be a single path segment")
    return stripped


def _normalize_relative_path(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must not contain empty, '.' or '..' segments")
    return str(path)


def _resolve_or_inferred(
    *,
    supplied: str | None,
    inferred: str | None,
    field_name: str,
    review_item_id: str,
) -> str:
    if supplied is None and inferred is None:
        raise ValueError("Missing {} for review item {}".format(field_name, review_item_id))
    if supplied is not None and inferred is not None and supplied != inferred:
        raise ValueError(
            "Conflicting {} for review item {}: supplied='{}' inferred='{}'".format(
                field_name,
                review_item_id,
                supplied,
                inferred,
            )
        )
    return supplied if supplied is not None else inferred  # type: ignore[return-value]


def _validate_relative_path_shape(
    *,
    source_relative_path: str,
    source_collection_name: str,
    source_category_dir: str,
    review_item_id: str,
) -> None:
    prefixed_with_collection = f"{source_collection_name}/{source_category_dir}/"
    if source_relative_path.startswith(prefixed_with_collection):
        raise ValueError(
            "source_relative_path for review item {} must be relative to '{}' only; "
            "received collection-prefixed path '{}'".format(
                review_item_id,
                source_category_dir,
                source_relative_path,
            )
        )
    prefixed_with_category = f"{source_category_dir}/"
    if source_relative_path.startswith(prefixed_with_category):
        raise ValueError(
            "source_relative_path for review item {} must be relative to '{}' only; "
            "received category-prefixed path '{}'".format(
                review_item_id,
                source_category_dir,
                source_relative_path,
            )
        )


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
    review_item_id: str,
) -> str:
    payload = f"{prediction_run_id}|{review_session_id}|{review_item_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

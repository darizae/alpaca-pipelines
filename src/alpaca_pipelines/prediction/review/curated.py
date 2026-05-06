"""Collection-native curated prediction-review materialization."""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpaca_pipelines.datasets.audio_utils import extract_segment
from alpaca_pipelines.io_utils import read_json, write_json
from alpaca_pipelines.prediction.review.config import PredictionReviewSessionManifest
from alpaca_pipelines.runs.manager import RunManager

CURATED_SOURCE_TYPE: Literal["manual_review_curated"] = "manual_review_curated"
LEGACY_CURATED_ROOT_DIRNAME = "_curated_prediction_examples"
CURATED_MANIFEST_FILENAME = "manifest.json"
CURATED_TARGET_CATEGORY = "hums_curated_manual_review"
CURATED_NOISE_CATEGORY = "curated_noise_segmented"
CURATED_CATEGORY_NAMES = (CURATED_TARGET_CATEGORY, CURATED_NOISE_CATEGORY)


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
    legacy_snippet_wav_path: str | None = None


class _LegacyMigrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_roots: dict[str, str]
    manifest_paths: list[str]
    created_count: int
    updated_count: int
    skipped_count: int
    total_items: int
    source_recording_keys: list[str]


def curated_categories_root(
    *,
    collection_root: Path,
    destination_root: Path | None = None,
) -> Path:
    return destination_root if destination_root is not None else collection_root


def legacy_curated_sources_root(
    *,
    datasets_root: Path,
) -> Path:
    return datasets_root / LEGACY_CURATED_ROOT_DIRNAME


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

    root = curated_categories_root(
        collection_root=collection_root, destination_root=destination_root
    )
    root.mkdir(parents=True, exist_ok=True)

    result = _write_materialized_groups(
        root=root,
        prediction_run_id=prediction_run_id,
        review_session_id=review_session_id,
        materialization_items=materialization_items,
    )
    by_label = {"target": 0, "noise": 0}
    for item in materialization_items:
        by_label[item.label] += 1
    return {
        "category_roots": result.category_roots,
        "category_names": sorted(result.category_roots),
        "manifest_paths": result.manifest_paths,
        "prediction_run_id": prediction_run_id,
        "review_session_id": review_session_id,
        "counts_by_label": by_label,
        "created_count": result.created_count,
        "updated_count": result.updated_count,
        "skipped_count": result.skipped_count,
        "total_items": result.total_items,
        "source_recording_keys": result.source_recording_keys,
    }


def list_curated_prediction_categories(
    *,
    collection_root: Path,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    root = curated_categories_root(
        collection_root=collection_root, destination_root=destination_root
    )
    if not root.exists():
        return {
            "category_roots": {
                category_name: str(root) for category_name in CURATED_CATEGORY_NAMES
            },
            "category_names": list(CURATED_CATEGORY_NAMES),
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

    for collection_name, category_name, manifest_path in _iter_category_manifest_paths(root):
        try:
            raw_payload = read_json(manifest_path)
            if not isinstance(raw_payload, dict):
                raise ValueError("Expected JSON object")
            manifest = CuratedPredictionSourceManifest.model_validate(raw_payload)
        except Exception as exc:
            warnings.append(f"Invalid manifest {manifest_path}: {exc}")
            continue

        counts_by_collection[collection_name] += len(manifest.items)
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
                "collection_name": collection_name,
                "category_name": category_name,
                "prediction_run_id": manifest.prediction_run_id,
                "review_session_id": manifest.review_session_id,
                "source_recording_key": manifest.source_recording_key,
                "n_items": len(manifest.items),
                "counts_by_label": {
                    "target": sum(1 for item in manifest.items if item.label == "target"),
                    "noise": sum(1 for item in manifest.items if item.label == "noise"),
                },
            }
        )

    return {
        "category_roots": {category_name: str(root) for category_name in CURATED_CATEGORY_NAMES},
        "category_names": list(CURATED_CATEGORY_NAMES),
        "manifests": manifests,
        "counts_by_collection": dict(sorted(counts_by_collection.items())),
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_provenance_type": dict(sorted(counts_by_provenance_type.items())),
        "warnings": warnings,
    }


def list_curated_prediction_sources(
    *,
    collection_root: Path,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    return list_curated_prediction_categories(
        collection_root=collection_root,
        destination_root=destination_root,
    )


def migrate_legacy_curated_prediction_sources(
    *,
    collection_root: Path,
    datasets_root: Path,
    destination_root: Path | None = None,
    remove_legacy_root: bool = False,
) -> dict[str, Any]:
    legacy_root = legacy_curated_sources_root(datasets_root=datasets_root)
    if not legacy_root.exists():
        root = curated_categories_root(
            collection_root=collection_root, destination_root=destination_root
        )
        return {
            "legacy_root": str(legacy_root),
            "removed_legacy_root": False,
            "category_roots": {
                category_name: str(root) for category_name in CURATED_CATEGORY_NAMES
            },
            "category_names": list(CURATED_CATEGORY_NAMES),
            "manifest_paths": [],
            "created_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "total_items": 0,
            "source_recording_keys": [],
        }

    root = curated_categories_root(
        collection_root=collection_root, destination_root=destination_root
    )
    root.mkdir(parents=True, exist_ok=True)

    materialization_items: list[_MaterializationItem] = []
    prediction_run_id = "migrated"
    review_session_id = "legacy_migration"
    for manifest_path in sorted(legacy_root.rglob(CURATED_MANIFEST_FILENAME)):
        raw_payload = read_json(manifest_path)
        if not isinstance(raw_payload, dict):
            raise ValueError("Expected JSON object in manifest: {}".format(manifest_path))
        manifest = CuratedPredictionSourceManifest.model_validate(raw_payload)
        prediction_run_id = manifest.prediction_run_id
        review_session_id = manifest.review_session_id
        for item in manifest.items:
            materialization_items.append(
                _MaterializationItem(
                    curated_example_id=item.curated_example_id,
                    review_item_id=item.review_item_id,
                    source_audio_file=item.source_audio_file,
                    start_s=item.start_s,
                    end_s=item.end_s,
                    label=item.label,
                    detection_index=item.detection_index,
                    detection_score=item.detection_score,
                    source_collection_name=item.source_collection_name,
                    source_category_dir=item.source_category_dir,
                    source_relative_path=item.source_relative_path,
                    source_display_path=item.source_display_path,
                    source_recording_key=item.source_recording_key,
                    payload_json=item.payload_json,
                    legacy_snippet_wav_path=item.snippet_wav_path,
                )
            )

    result = _write_materialized_groups(
        root=root,
        prediction_run_id=prediction_run_id,
        review_session_id=review_session_id,
        materialization_items=materialization_items,
        legacy_manifest_root=legacy_root,
    )

    removed_legacy_root = False
    if remove_legacy_root and legacy_root.exists():
        shutil.rmtree(legacy_root)
        removed_legacy_root = True

    return {
        "legacy_root": str(legacy_root),
        "removed_legacy_root": removed_legacy_root,
        "category_roots": result.category_roots,
        "category_names": sorted(result.category_roots),
        "manifest_paths": result.manifest_paths,
        "created_count": result.created_count,
        "updated_count": result.updated_count,
        "skipped_count": result.skipped_count,
        "total_items": result.total_items,
        "source_recording_keys": result.source_recording_keys,
    }


def _write_materialized_groups(
    *,
    root: Path,
    prediction_run_id: str,
    review_session_id: str,
    materialization_items: list[_MaterializationItem],
    legacy_manifest_root: Path | None = None,
) -> _LegacyMigrationResult:
    groups: dict[
        tuple[str, str, str, str, str, str, str],
        list[_MaterializationItem],
    ] = defaultdict(list)
    for item in materialization_items:
        category_name = _category_name_for_label(item.label)
        key = (
            item.source_collection_name,
            category_name,
            item.source_category_dir,
            item.source_relative_path,
            item.source_display_path,
            item.source_recording_key,
            item.source_audio_file,
        )
        groups[key].append(item)

    manifests_created: list[str] = []
    created_count = 0
    updated_count = 0
    skipped_count = 0
    source_recording_keys: set[str] = set()
    category_roots: dict[str, str] = {}

    for (
        collection_name,
        category_name,
        source_category_dir,
        source_relative_path,
        source_display_path,
        source_recording_key,
        source_audio_file,
    ), grouped_items in sorted(groups.items()):
        recording_dir = root / collection_name / category_name / source_recording_key
        recording_dir.mkdir(parents=True, exist_ok=True)
        manifest_out_path = recording_dir / CURATED_MANIFEST_FILENAME
        existing_items_by_id = _load_existing_items(manifest_out_path)
        new_items: list[CuratedPredictionSourceItem] = []

        for item in sorted(grouped_items, key=lambda value: value.review_item_id):
            source_recording_keys.add(item.source_recording_key)
            curated_example_id = item.curated_example_id or _build_curated_example_id(
                prediction_run_id=prediction_run_id,
                review_session_id=review_session_id,
                review_item_id=item.review_item_id,
            )
            snippet_filename = f"{item.label}_{curated_example_id}.wav"
            snippet_path = recording_dir / snippet_filename
            previous = existing_items_by_id.get(curated_example_id)
            needs_update = (
                previous is None
                or previous.label != item.label
                or previous.start_s != item.start_s
                or previous.end_s != item.end_s
                or previous.source_recording_key != item.source_recording_key
                or previous.source_audio_file != item.source_audio_file
                or previous.snippet_wav_path != str(snippet_path)
                or not snippet_path.is_file()
            )
            if needs_update:
                duration_s = _materialize_or_copy_snippet(
                    item=item,
                    snippet_path=snippet_path,
                    previous=previous,
                    legacy_manifest_root=legacy_manifest_root,
                )
                if previous is None:
                    created_count += 1
                else:
                    updated_count += 1
            else:
                if previous is None:
                    raise AssertionError("previous item must exist when snippet update is skipped")
                duration_s = previous.duration_s
                skipped_count += 1

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
        category_roots[category_name] = str(root)

    return _LegacyMigrationResult(
        category_roots=category_roots,
        manifest_paths=manifests_created,
        created_count=created_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
        total_items=len(materialization_items),
        source_recording_keys=sorted(source_recording_keys),
    )


def _materialize_or_copy_snippet(
    *,
    item: _MaterializationItem,
    snippet_path: Path,
    previous: CuratedPredictionSourceItem | None,
    legacy_manifest_root: Path | None,
) -> float:
    if (
        legacy_manifest_root is not None
        and item.legacy_snippet_wav_path
        and Path(item.legacy_snippet_wav_path).is_file()
    ):
        legacy_snippet_path = Path(item.legacy_snippet_wav_path)
        if legacy_snippet_path.is_file():
            snippet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_snippet_path, snippet_path)
            return item.end_s - item.start_s
    return extract_segment(
        source_path=Path(item.source_audio_file),
        start_s=item.start_s,
        end_s=item.end_s,
        destination_path=snippet_path,
    )


def _iter_category_manifest_paths(root: Path) -> list[tuple[str, str, Path]]:
    discovered: list[tuple[str, str, Path]] = []
    for collection_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for category_name in CURATED_CATEGORY_NAMES:
            category_dir = collection_dir / category_name
            if not category_dir.is_dir():
                continue
            for manifest_path in sorted(category_dir.rglob(CURATED_MANIFEST_FILENAME)):
                discovered.append((collection_dir.name, category_name, manifest_path))
    return discovered


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
    if not review_item_id:
        raise ValueError("review_item_id is required for curated materialization")
    if not source_collection_name:
        raise ValueError("source_collection_name is required for curated materialization")
    if not source_category_dir:
        raise ValueError("source_category_dir is required for curated materialization")
    if not source_relative_path:
        raise ValueError("source_relative_path is required for curated materialization")
    if not source_audio_file:
        raise ValueError("source_audio_file is required for curated materialization")
    if end_s <= start_s:
        raise ValueError(
            "Curated item {} has invalid bounds {}-{}".format(review_item_id, start_s, end_s)
        )

    relative_path = PurePosixPath(source_relative_path)
    if relative_path.is_absolute():
        raise ValueError("source_relative_path must be relative: {}".format(source_relative_path))
    if source_collection_name in relative_path.parts:
        raise ValueError(
            "source_relative_path must not include collection prefix: {}".format(
                source_relative_path
            )
        )

    display_path = "{}/{}/{}".format(
        source_collection_name,
        source_category_dir,
        relative_path.as_posix(),
    )
    resolved_audio_path = Path(source_audio_file)
    if not resolved_audio_path.is_file():
        candidate = collection_root / source_collection_name / source_category_dir / relative_path
        if not candidate.is_file():
            raise FileNotFoundError(
                "Source audio file not found for curated materialization: {}".format(
                    source_audio_file
                )
            )
        resolved_audio_path = candidate

    payload_source_recording_key = None
    if payload_json and isinstance(payload_json.get("source_recording_key"), str):
        payload_source_recording_key = str(payload_json["source_recording_key"])
    normalized_recording_key = (
        source_recording_key or payload_source_recording_key or resolved_audio_path.stem
    )
    if not normalized_recording_key:
        raise ValueError(
            "Unable to derive source_recording_key for curated item {}".format(review_item_id)
        )

    return _MaterializationItem(
        curated_example_id=curated_example_id,
        review_item_id=review_item_id,
        source_audio_file=str(resolved_audio_path),
        start_s=start_s,
        end_s=end_s,
        label=label,
        detection_index=detection_index,
        detection_score=detection_score,
        source_collection_name=source_collection_name,
        source_category_dir=source_category_dir,
        source_relative_path=relative_path.as_posix(),
        source_display_path=display_path,
        source_recording_key=normalized_recording_key,
        payload_json=payload_json,
    )


def _category_name_for_label(label: Literal["target", "noise"]) -> str:
    return CURATED_TARGET_CATEGORY if label == "target" else CURATED_NOISE_CATEGORY


def _build_curated_example_id(
    *,
    prediction_run_id: str,
    review_session_id: str,
    review_item_id: str,
) -> str:
    payload = "{}:{}:{}".format(prediction_run_id, review_session_id, review_item_id).encode(
        "utf-8"
    )
    return hashlib.sha1(payload).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

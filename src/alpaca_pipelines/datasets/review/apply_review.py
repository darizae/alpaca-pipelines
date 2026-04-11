from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from alpaca_pipelines.datasets.contracts import Manifest, SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.io_utils import write_json
from alpaca_pipelines.datasets.paths import CORRECTIONS_APPLIED_FILENAME, REVIEW_DIR, SNIPPETS_DIR

ReviewAnnotation = Literal["target", "noise", "discard"]


def _validate_reclassification(snippet: SnippetEntry, new_classification: str) -> None:
    if new_classification == "target" and snippet.classification == "noise":
        if snippet.quality is None or snippet.subject_id is None or snippet.recording_date is None:
            raise ValueError(
                f"Cannot reclassify noise→target for uid={snippet.uid}: "
                f"missing required target metadata "
                f"(quality={snippet.quality}, subject_id={snippet.subject_id}, "
                f"recording_date={snippet.recording_date})"
            )


def _validate_snippet_filename(filename: str) -> None:
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Snippet filename contains path separator: {filename!r}")
    if ".." in filename:
        raise ValueError(f"Snippet filename contains traversal: {filename!r}")
    if filename != Path(filename).name:
        raise ValueError(f"Snippet filename is not a plain basename: {filename!r}")


def _reclassified_filename(snippet: SnippetEntry, new_classification: str) -> str:
    if new_classification == "target":
        if snippet.quality is None:
            raise ValueError(
                f"Cannot generate target filename for uid={snippet.uid}: quality is None"
            )
        label = f"Q{snippet.quality}"
    else:
        label = "lowq" if snippet.source_type == "low_quality_hum" else "bg"
    uid_part = f"{snippet.uid:06d}"
    return f"{new_classification}-{label}_{uid_part}_{snippet.collection}.wav"


def apply_review_table(
    dataset_dir: Path,
    manifest: Manifest,
    target_review_table_path: Path,
    noise_review_table_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> tuple[Manifest, int, int]:
    target_snippets = _sorted_snippets_by_class(manifest, "target")
    noise_snippets = _sorted_snippets_by_class(manifest, "noise")

    target_labels = _parse_review_labels(
        review_table_path=target_review_table_path,
        class_snippets=target_snippets,
        fs=fs,
    )
    noise_labels = _parse_review_labels(
        review_table_path=noise_review_table_path,
        class_snippets=noise_snippets,
        fs=fs,
    )

    overlap = set(target_labels) & set(noise_labels)
    if overlap:
        raise ValueError(
            "Review tables map some snippets twice across target/noise uploads. "
            f"First 10 UIDs: {sorted(overlap)[:10]}"
        )

    label_by_uid: dict[int, ReviewAnnotation] = {**target_labels, **noise_labels}
    manifest_uids = {snippet.uid for snippet in manifest.snippets}
    missing_from_tables = manifest_uids - set(label_by_uid)
    if missing_from_tables:
        raise ValueError(
            f"Review table is incomplete: {len(missing_from_tables)} manifest UIDs missing. "
            f"First 10: {sorted(missing_from_tables)[:10]}"
        )
    unknown_uids = set(label_by_uid) - manifest_uids
    if unknown_uids:
        raise ValueError(f"Review table references unknown UIDs: {sorted(unknown_uids)[:10]}")

    uid_to_snippet: dict[int, SnippetEntry] = {s.uid: s for s in manifest.snippets}
    corrections: list[dict[str, object]] = []
    discarded_uids: set[int] = set()
    updated_snippets: dict[int, SnippetEntry] = {}
    snippets_dir = dataset_dir / SNIPPETS_DIR

    for uid, annotation in label_by_uid.items():
        snippet = uid_to_snippet[uid]
        if annotation == "discard":
            discarded_uids.add(uid)
            corrections.append(
                {
                    "uid": uid,
                    "action": "discard",
                    "previous_classification": snippet.classification,
                }
            )
            continue

        if annotation != snippet.classification:
            _validate_reclassification(snippet, annotation)
            _validate_snippet_filename(snippet.filename)

            new_filename = _reclassified_filename(snippet, annotation)
            old_wav = snippets_dir / snippet.filename
            new_wav = snippets_dir / new_filename
            if not fs.exists(old_wav):
                raise FileNotFoundError(
                    f"Cannot rename snippet uid={uid}: source file missing: {old_wav}"
                )
            if fs.exists(new_wav):
                raise FileExistsError(
                    f"Cannot rename snippet uid={uid}: destination already exists: {new_wav}"
                )

            fs.rename(old_wav, new_wav)

            updated_snippets[uid] = snippet.model_copy(
                update={
                    "classification": annotation,
                    "filename": new_filename,
                    "review_status": "approved",
                }
            )
            corrections.append(
                {
                    "uid": uid,
                    "action": "reclassify",
                    "previous_classification": snippet.classification,
                    "new_classification": annotation,
                    "old_filename": snippet.filename,
                    "new_filename": new_filename,
                }
            )
        else:
            updated_snippets[uid] = snippet.model_copy(update={"review_status": "approved"})

    retained_snippets: list[SnippetEntry] = []
    for snippet in manifest.snippets:
        if snippet.uid in discarded_uids:
            _validate_snippet_filename(snippet.filename)
            discarded_wav = snippets_dir / snippet.filename
            if not fs.exists(discarded_wav):
                raise FileNotFoundError(
                    f"Cannot discard snippet uid={snippet.uid}: file missing: {discarded_wav}"
                )
            fs.unlink(discarded_wav)
            continue

        if snippet.uid in updated_snippets:
            retained_snippets.append(updated_snippets[snippet.uid])
        else:
            retained_snippets.append(snippet)

    n_reclassified = sum(1 for c in corrections if c["action"] == "reclassify")
    n_discarded = len(discarded_uids)

    updated_manifest = manifest.model_copy(
        update={
            "snippets": retained_snippets,
            "meta": manifest.meta.model_copy(
                update={
                    "n_snippets": len(retained_snippets),
                    "n_target": sum(1 for s in retained_snippets if s.classification == "target"),
                    "n_noise": sum(1 for s in retained_snippets if s.classification == "noise"),
                }
            ),
        }
    )

    corrections_log = {
        "applied_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_review_table_path": str(target_review_table_path),
        "noise_review_table_path": str(noise_review_table_path),
        "n_reclassified": n_reclassified,
        "n_discarded": n_discarded,
        "corrections": corrections,
    }
    corrections_path = dataset_dir / REVIEW_DIR / CORRECTIONS_APPLIED_FILENAME
    write_json(corrections_path, corrections_log, fs)

    return updated_manifest, n_reclassified, n_discarded


def _sorted_snippets_by_class(
    manifest: Manifest,
    classification: Literal["target", "noise"],
) -> list[SnippetEntry]:
    return sorted(
        [snippet for snippet in manifest.snippets if snippet.classification == classification],
        key=lambda snippet: (snippet.collection, snippet.uid),
    )


def _parse_review_labels(
    *,
    review_table_path: Path,
    class_snippets: list[SnippetEntry],
    fs: FileSystem,
) -> dict[int, ReviewAnnotation]:
    content = fs.read_text(review_table_path)
    review_table = pd.read_csv(io.StringIO(content), sep="\t")
    columns = set(review_table.columns)
    has_uid = "uid" in columns
    has_selection = "Selection" in columns
    if not has_uid and not has_selection:
        raise ValueError(
            f"Review table {review_table_path} must include either 'uid' or 'Selection'"
        )
    has_review_label = "review_label" in columns
    has_sound_type = "Sound_type" in columns
    if not has_review_label and not has_sound_type:
        raise ValueError(
            f"Review table {review_table_path} must include either 'review_label' or 'Sound_type'"
        )

    if not class_snippets:
        if len(review_table.index) == 0:
            return {}
        raise ValueError(
            f"Review table {review_table_path} contains rows, but this class has no snippets"
        )

    selection_to_uid = {index + 1: snippet.uid for index, snippet in enumerate(class_snippets)}
    allowed_uids = set(selection_to_uid.values())
    parsed: dict[int, ReviewAnnotation] = {}

    for row_number, (_, row) in enumerate(review_table.iterrows(), start=2):
        uid_value = _cell_text(row, "uid") if has_uid else ""
        selection_value = _cell_text(row, "Selection") if has_selection else ""
        if uid_value:
            uid = _parse_int(uid_value, field_name="uid", row_number=row_number)
        elif selection_value:
            selection = _parse_int(
                selection_value,
                field_name="Selection",
                row_number=row_number,
            )
            resolved_uid = selection_to_uid.get(selection)
            if resolved_uid is None:
                raise ValueError(
                    f"Review table row {row_number} has Selection {selection} outside valid range "
                    f"1..{len(selection_to_uid)}"
                )
            uid = resolved_uid
        else:
            raise ValueError(f"Review table row {row_number} is missing both uid and Selection")

        if uid not in allowed_uids:
            raise ValueError(
                "Review table {} references uid {} not present in this class subset".format(
                    review_table_path,
                    uid,
                )
            )

        raw_label = _cell_text(row, "review_label") if has_review_label else ""
        if raw_label == "":
            raw_label = _cell_text(row, "Sound_type")
        annotation = _normalize_review_label(
            raw_label,
            row_number=row_number,
        )

        existing = parsed.get(uid)
        if existing is not None and existing != annotation:
            raise ValueError(
                "Review table contains conflicting labels for uid {}: {} vs {}".format(
                    uid,
                    existing,
                    annotation,
                )
            )
        parsed[uid] = annotation

    missing = allowed_uids - set(parsed)
    if missing:
        raise ValueError(
            "Review table {} is incomplete: {} snippet rows are missing. First 10 UIDs: {}".format(
                review_table_path,
                len(missing),
                sorted(missing)[:10],
            )
        )
    return parsed


def _cell_text(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _parse_int(value: str, *, field_name: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Review table row {row_number} has invalid {field_name}: {value!r}"
        ) from exc


def _normalize_review_label(value: str, *, row_number: int) -> ReviewAnnotation:
    normalized = value.strip().lower()
    if normalized in {"1", "target"}:
        return "target"
    if normalized in {"0", "noise"}:
        return "noise"
    if normalized == "discard":
        return "discard"
    if normalized == "":
        raise ValueError(f"Review table row {row_number} has an empty label")
    raise ValueError(
        f"Review table row {row_number} has unknown label {value!r}. "
        "Valid values: 1, 0, target, noise, discard"
    )

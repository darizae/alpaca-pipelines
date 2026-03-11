from __future__ import annotations

import io
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from alpaca_pipelines.datasets.contracts import VALID_REVIEW_ANNOTATIONS, Manifest, SnippetEntry
from alpaca_pipelines.datasets.fs import _DEFAULT_FS, FileSystem
from alpaca_pipelines.datasets.io_utils import write_json
from alpaca_pipelines.datasets.paths import CORRECTIONS_APPLIED_FILENAME, REVIEW_DIR, SNIPPETS_DIR


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
    review_table_path: Path,
    fs: FileSystem = _DEFAULT_FS,
) -> tuple[Manifest, int, int]:
    content = fs.read_text(review_table_path)
    review_table = pd.read_csv(io.StringIO(content), sep="\t")

    required_columns = {"uid", "Sound_type"}
    missing_columns = required_columns - set(review_table.columns)
    if missing_columns:
        raise ValueError(f"Review table missing columns: {missing_columns}")

    manifest_uids = {s.uid for s in manifest.snippets}
    table_uids: list[int] = [int(row["uid"]) for _, row in review_table.iterrows()]

    uid_counts = Counter(table_uids)
    duplicates = {uid for uid, count in uid_counts.items() if count > 1}
    if duplicates:
        raise ValueError(f"Review table contains duplicate UIDs: {sorted(duplicates)}")

    table_uid_set = set(table_uids)
    missing_from_table = manifest_uids - table_uid_set
    if missing_from_table:
        raise ValueError(
            f"Review table is incomplete: {len(missing_from_table)} manifest UIDs missing. "
            f"First 10: {sorted(missing_from_table)[:10]}"
        )

    unknown_uids = table_uid_set - manifest_uids
    if unknown_uids:
        raise ValueError(f"Review table references unknown UIDs: {sorted(unknown_uids)[:10]}")

    uid_to_snippet: dict[int, SnippetEntry] = {s.uid: s for s in manifest.snippets}
    corrections: list[dict[str, object]] = []
    discarded_uids: set[int] = set()
    updated_snippets: dict[int, SnippetEntry] = {}
    snippets_dir = dataset_dir / SNIPPETS_DIR

    for _, row in review_table.iterrows():
        uid = int(row["uid"])
        annotation = str(row["Sound_type"]).strip()

        if annotation not in VALID_REVIEW_ANNOTATIONS:
            raise ValueError(
                f"Unknown review annotation '{annotation}' for uid={uid}. "
                f"Valid values: {sorted(VALID_REVIEW_ANNOTATIONS)}"
            )

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
        "review_table_path": str(review_table_path),
        "n_reclassified": n_reclassified,
        "n_discarded": n_discarded,
        "corrections": corrections,
    }
    corrections_path = dataset_dir / REVIEW_DIR / CORRECTIONS_APPLIED_FILENAME
    write_json(corrections_path, corrections_log, fs)

    return updated_manifest, n_reclassified, n_discarded

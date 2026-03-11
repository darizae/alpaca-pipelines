from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpaca_pipelines.collections.contracts import IdentityMap
from alpaca_pipelines.collections.fs import FileSystem
from alpaca_pipelines.collections.parsing.normalizers import normalize_labelled_clip_filename
from alpaca_pipelines.collections.parsing.parse import normalize_segmented_hum_filename
from alpaca_pipelines.collections.paths import (
    LEGACY_CLIPS_DIR_NAMES,
    LEGACY_HUMS_DIR_NAMES,
    CategoryNames,
    find_category_dir,
)


@dataclass(frozen=True)
class RenameOp:
    src: str
    dst: str


@dataclass(frozen=True)
class ClipAuditRow:
    old_path: str
    new_path: str
    subject_token_original: str
    subject_id: str
    date_yyyymmdd: str
    time_hhmmss: str | None
    note: str | None


@dataclass(frozen=True)
class HumAuditRow:
    old_path: str
    new_path: str


@dataclass(frozen=True)
class RenamePlan:
    root: str
    ops: list[RenameOp]
    audit: dict[str, Any]


def _detect_plan_collisions(ops: list[RenameOp]) -> None:
    dst_to_src: dict[str, str] = {}
    seen_src: set[str] = set()

    for op in ops:
        if op.src in seen_src:
            raise ValueError(f"Duplicate src in plan: {op.src}")
        seen_src.add(op.src)

        existing = dst_to_src.get(op.dst)
        if existing is not None and existing != op.src:
            raise ValueError(
                f"Destination collision in plan: {op.dst} from {existing} and {op.src}"
            )
        dst_to_src[op.dst] = op.src


def plan_renames_for_collection(
    collection_dir: Path,
    identity_map: IdentityMap,
    category_names: CategoryNames,
    fs: FileSystem,
) -> tuple[list[RenameOp], list[ClipAuditRow], list[HumAuditRow]]:
    clips_src = find_category_dir(
        collection_dir, category_names.clips_labelled, LEGACY_CLIPS_DIR_NAMES, fs
    )
    hums_src = find_category_dir(
        collection_dir, category_names.hums_segmented, LEGACY_HUMS_DIR_NAMES, fs
    )
    if clips_src is None or hums_src is None:
        raise FileNotFoundError(f"Missing category dirs under {collection_dir}")

    clip_ops: list[RenameOp] = []
    hum_ops: list[RenameOp] = []
    dir_ops: list[RenameOp] = []

    clip_audit: list[ClipAuditRow] = []
    hum_audit: list[HumAuditRow] = []

    for clip_path in fs.rglob_wavs(clips_src):
        normalized = normalize_labelled_clip_filename(clip_path, identity_map, fs)
        new_filename = normalized.to_filename()

        if clip_path.name == new_filename:
            continue

        dst_path = clip_path.with_name(new_filename)
        clip_ops.append(RenameOp(src=str(clip_path), dst=str(dst_path)))

        clip_audit.append(
            ClipAuditRow(
                old_path=str(clip_path.relative_to(collection_dir)),
                new_path=str(dst_path.relative_to(collection_dir)),
                subject_token_original=normalized.subject_alias,
                subject_id=normalized.subject_id,
                date_yyyymmdd=normalized.date_yyyymmdd,
                time_hhmmss=normalized.time_hhmmss,
                note=normalized.note,
            )
        )

    for hum_path in fs.rglob_wavs(hums_src):
        canonical_name = normalize_segmented_hum_filename(hum_path)  # validates only
        if hum_path.name == canonical_name:
            continue

        dst_path = hum_path.with_name(canonical_name)
        hum_ops.append(RenameOp(src=str(hum_path), dst=str(dst_path)))
        hum_audit.append(
            HumAuditRow(
                old_path=str(hum_path.relative_to(collection_dir)),
                new_path=str(dst_path.relative_to(collection_dir)),
            )
        )

    if clips_src.name != category_names.clips_labelled:
        dir_ops.append(
            RenameOp(src=str(clips_src), dst=str(collection_dir / category_names.clips_labelled))
        )

    if hums_src.name != category_names.hums_segmented:
        dir_ops.append(
            RenameOp(src=str(hums_src), dst=str(collection_dir / category_names.hums_segmented))
        )

    ops = [*clip_ops, *hum_ops, *dir_ops]
    _detect_plan_collisions(ops)

    return ops, clip_audit, hum_audit

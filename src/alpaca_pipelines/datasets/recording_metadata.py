from __future__ import annotations

from datetime import datetime, timedelta

from alpaca_pipelines.datasets.contracts import SnippetEntry
from alpaca_pipelines.recordings import SourceRecording, recording_track_point_at_offset


def with_recording_window(
    snippet: SnippetEntry,
    recording: SourceRecording | None,
    *,
    start_offset_s: float | None,
    end_offset_s: float | None,
) -> SnippetEntry:
    if recording is None or start_offset_s is None or end_offset_s is None:
        return snippet

    midpoint_offset = max(0.0, (start_offset_s + end_offset_s) / 2.0)
    midpoint = recording_track_point_at_offset(recording, midpoint_offset)

    return snippet.model_copy(
        update={
            "source_recording_key": recording.key,
            "source_recording_start_s": round(start_offset_s, 4),
            "source_recording_end_s": round(end_offset_s, 4),
            "snippet_started_at": _offset_iso(recording.start_time, start_offset_s),
            "snippet_ended_at": _offset_iso(recording.start_time, end_offset_s),
            "snippet_midpoint_latitude": midpoint.latitude if midpoint is not None else None,
            "snippet_midpoint_longitude": midpoint.longitude if midpoint is not None else None,
            "snippet_gps_status": midpoint.gps_status if midpoint is not None else None,
        }
    )


def _offset_iso(start_time: str | None, offset_seconds: float) -> str | None:
    if start_time is None:
        return None
    started_at = datetime.fromisoformat(start_time)
    return (started_at + timedelta(seconds=offset_seconds)).isoformat(timespec="milliseconds")

from __future__ import annotations

import re

CANONICAL_CLIP_RE = re.compile(
    r"^(?P<subject>[A-Za-z0-9]+)_(?P<date>\d{8})"
    r"(?:_(?P<time>\d{6}))?"
    r"(?:_(?P<note>.+?))?"
    r"\.wav_(?P<clip_start>\d+)_(?P<clip_end>\d+)\.wav$"
)

COL1_CLIP_RE = re.compile(
    r"^(?P<subject>[A-Za-z0-9]+)_(?P<date>\d{8})(?:_(?P<tag>[^_]+))?_cut\.wav_"
    r"(?P<clip_start>\d+)_(?P<clip_end>\d+)\.wav$"
)

COL2_LABELLED_DT_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?:[ _](?P<note>.+?))?\.wav$")

COL2_LABELLED_SUBJECT_DT_RE = re.compile(
    r"^(?P<subject>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})(?:[ _](?P<note>.+?))?\.wav$"
)

COL2_LABELLED_SUBJECT_DATE_RE = re.compile(
    r"^(?P<subject>[A-Za-z0-9]+)_(?P<date>\d{8})(?:[ _](?P<note>.+?))?\.wav$"
)

CANONICAL_HUM_RE = re.compile(
    r"^(?P<clip>.+?\.wav_\d+_\d+\.wav)_(?P<hum_start>\d+(?:\.\d+)?)_(?P<hum_end>\d+(?:\.\d+)?)Q(?P<q>\d)\.wav$"
)

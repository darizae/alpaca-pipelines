from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class CanonicalSubject(BaseModel):
    display_name: str


class IdentityMap(BaseModel):
    canonical: dict[str, CanonicalSubject]
    aliases: dict[str, str]

    def normalize_subject(self, token: str) -> str:
        if token in self.canonical:
            return token
        if token in self.aliases:
            return self.aliases[token]
        lowered = token.lower()
        canonical_lowered = {key.lower(): key for key in self.canonical}
        if lowered in canonical_lowered:
            return canonical_lowered[lowered]
        if lowered in self.aliases:
            return self.aliases[lowered]
        raise ValueError(f"Unknown subject token '{token}'. Add it to identity map aliases.")


def load_identity_map(path: Path) -> IdentityMap:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return IdentityMap.model_validate(data)

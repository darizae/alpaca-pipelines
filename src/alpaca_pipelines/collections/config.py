from __future__ import annotations

from pydantic import BaseModel, Field


class StandardizerConfig(BaseModel):
    min_source_quality_to_keep: int | None = Field(
        default=None,
        description="If set, keep only hums with Q >= this value. If None, keep all.",
    )

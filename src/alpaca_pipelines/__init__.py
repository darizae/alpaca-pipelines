"""Alpaca Pipelines package."""

from __future__ import annotations

from typing import Any

__all__ = ["PipelineAPI"]


def __getattr__(name: str) -> Any:
    if name == "PipelineAPI":
        from alpaca_pipelines.api import PipelineAPI

        return PipelineAPI
    raise AttributeError(name)

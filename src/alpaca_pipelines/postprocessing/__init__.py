"""Post-processing utilities for pipeline outputs."""

from alpaca_pipelines.postprocessing.executor import (
    aggregate_evaluation_results,
    export_detections_to_selection_table,
)

__all__ = [
    "aggregate_evaluation_results",
    "export_detections_to_selection_table",
]

"""Convert survey plat sheets into DXF drawings."""

import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorjuju.calibrate import Scale, ScaleCalibrationError, calibrate_scale
    from vectorjuju.export import write_outputs
    from vectorjuju.pipeline import UnsupportedInputError, convert
    from vectorjuju.text import (
        BoundCall,
        ParsedCall,
        TextItem,
        bind_calls,
        extract_text,
        page_items,
        parse_call,
        text_mask,
    )
    from vectorjuju.tracing import Run, trace_runs

__version__ = importlib.metadata.version("vectorjuju")

_CALIBRATE_NAMES = {"Scale", "ScaleCalibrationError", "calibrate_scale"}
_EXPORT_NAMES = {"write_outputs"}
_PIPELINE_NAMES = {"UnsupportedInputError", "convert"}
_TRACING_NAMES = {"Run", "trace_runs"}
_TEXT_NAMES = {
    "BoundCall",
    "ParsedCall",
    "TextItem",
    "bind_calls",
    "extract_text",
    "page_items",
    "parse_call",
    "text_mask",
}

__all__ = [
    "BoundCall",
    "ParsedCall",
    "Run",
    "Scale",
    "ScaleCalibrationError",
    "TextItem",
    "UnsupportedInputError",
    "__version__",
    "bind_calls",
    "calibrate_scale",
    "convert",
    "extract_text",
    "page_items",
    "parse_call",
    "text_mask",
    "trace_runs",
    "write_outputs",
]


def __getattr__(name: str):
    """Import the pipeline/tracing/text/calibration APIs lazily: `import vectorjuju` must not need cv2/skimage/docling."""
    if name in _PIPELINE_NAMES:
        from vectorjuju import pipeline

        return getattr(pipeline, name)
    if name in _TRACING_NAMES:
        from vectorjuju import tracing

        return getattr(tracing, name)
    if name in _TEXT_NAMES:
        from vectorjuju import text

        return getattr(text, name)
    if name in _CALIBRATE_NAMES:
        from vectorjuju import calibrate

        return getattr(calibrate, name)
    if name in _EXPORT_NAMES:
        from vectorjuju import export

        return getattr(export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

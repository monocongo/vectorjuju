"""Convert survey plat sheets into DXF drawings."""

import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorjuju.calibrate import Scale, ScaleCalibrationError, calibrate_scale
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
    "__version__",
    "bind_calls",
    "calibrate_scale",
    "extract_text",
    "page_items",
    "parse_call",
    "text_mask",
    "trace_runs",
]


def __getattr__(name: str):
    """Import the tracing/text/calibration APIs lazily: `import vectorjuju` must not need cv2/skimage/docling."""
    if name in _TRACING_NAMES:
        from vectorjuju import tracing

        return getattr(tracing, name)
    if name in _TEXT_NAMES:
        from vectorjuju import text

        return getattr(text, name)
    if name in _CALIBRATE_NAMES:
        from vectorjuju import calibrate

        return getattr(calibrate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Convert survey plat sheets into DXF drawings."""

import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorjuju.tracing import Run, trace_runs

__version__ = importlib.metadata.version("vectorjuju")

__all__ = ["Run", "__version__", "trace_runs"]


def __getattr__(name: str):
    """Import the tracing API lazily: `import vectorjuju` must not need cv2/skimage."""
    if name in {"Run", "trace_runs"}:
        from vectorjuju import tracing

        return getattr(tracing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

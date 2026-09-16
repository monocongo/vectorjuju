"""Helpers shared by the run-level gates over the synthetic plat.

The full acceptance suite (issue #22) builds its geometry comparisons on
these; keep them free of tracing and DXF concerns.
"""

from __future__ import annotations

import math

import numpy as np

from vectorjuju.curves import CircleFit


def arc_midpoint(points: np.ndarray, fit: CircleFit) -> np.ndarray:
    """Point halfway along the fitted sweep from the run's start to its end."""
    centre = np.asarray(fit.center_px)
    start = math.atan2(points[0][1] - centre[1], points[0][0] - centre[0])
    end = math.atan2(points[-1][1] - centre[1], points[-1][0] - centre[0])
    if fit.clockwise:  # increasing raster angle turns clockwise on the page
        while end <= start:
            end += 2 * math.pi
    else:
        while end >= start:
            end -= 2 * math.pi
    angle = (start + end) / 2
    return centre + fit.radius_px * np.array([math.cos(angle), math.sin(angle)])

"""Fit and classify traced runs as straight lines or circular arcs.

A run is a circular arc when a least-squares circle through its points has a
residual below the fit gate and a radius at least half its chord; a run that
bows more than the tracing simplification tolerance but does not fit a circle
is still a curve, and the writer falls back to a SPLINE rather than pretend it
is straight. Everything else is a line.

Classification is the normalized circle residual, not chord deviation: the
prior art measured an L-corner's chord deviation *larger* than a true arc's
(`docs/prior-art/20-implementation-plan.md`, experiment 6), so no deviation
threshold separates them. The prior art's consistent-turn check is not
carried: it never decides the corner case, and a reversing trace already
shows up as a bad circle fit.

The fit is in raster px (top-left origin, y down). ``classify`` carries the
sweep direction so the DXF writer can emit the bulge on the side the trace
actually runs; ``px_to_cad`` flips y, so a run that sweeps clockwise on the
raster page also sweeps clockwise in CAD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.measure import CircleModel

from vectorjuju.tracing import _SIMPLIFY_PX, _scaled

# Max fit residual as a fraction of the fitted radius. Clean traced arcs land
# near 0.001; the prior art's L-corner measured ~13, so 0.03 separates them
# with two orders of magnitude of headroom, and it matches the suite's 3 %
# radius tolerance for the arcs the writer emits as ARC.
_MAX_RESIDUAL_RATIO = 0.03

Kind = Literal["line", "curve"]


@dataclass(frozen=True)
class CircleFit:
    """Least-squares circle through one run, in raster px (y down)."""

    center_px: tuple[float, float]
    radius_px: float
    max_residual_px: float
    clockwise: bool
    """Whether the run sweeps clockwise on the page; raster and CAD agree."""


def classify(points_px: ArrayLike, dpi: float) -> tuple[Kind, CircleFit | None]:
    """Classify one traced run, in raster px, as a line or a curve.

    ``("line", None)`` for a straight run, ``("curve", fit)`` for a circular
    arc (emit ARC), and ``("curve", None)`` for a bent run that does not fit a
    circle (emit the SPLINE fallback). ``dpi`` scales the line/curve bow gate
    the same way ``trace_runs`` scales its simplification tolerance.
    """
    points = np.asarray(points_px, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"points_px must be an (N, 2) array with N >= 2, got shape {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_px must be finite")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"dpi must be a positive finite number, got {dpi!r}")

    # The tracer simplified this run with _SIMPLIFY_PX at the reference DPI; a
    # bend below that tolerance is indistinguishable from straight linework.
    if _chord_deviation(points) <= _scaled(_SIMPLIFY_PX, dpi):
        return "line", None

    fit = _fit_circle(points)
    if (
        fit is not None
        and fit.radius_px >= 0.5 * _chord_length(points)  # a circle through the endpoints cannot be tighter
        and fit.max_residual_px <= _MAX_RESIDUAL_RATIO * fit.radius_px
    ):
        return "curve", fit
    return "curve", None


def _fit_circle(points: NDArray[np.float64]) -> CircleFit | None:
    """Least-squares circle, or None when the points cannot estimate one."""
    model = CircleModel.from_estimate(points)
    if not model:  # FailedEstimation: too few non-collinear points
        return None
    center = np.asarray(model.center, dtype=float)
    return CircleFit(
        center_px=(float(center[0]), float(center[1])),
        radius_px=float(model.radius),
        max_residual_px=float(np.abs(model.residuals(points)).max()),
        clockwise=_sweeps_clockwise(points, center),
    )


def _sweeps_clockwise(points: NDArray[np.float64], center: NDArray[np.float64]) -> bool:
    """Sign of the run's turning about the centre, in the raster's y-down axes.

    A positive cross product turns from +x towards +y, which is clockwise as
    the raster page is viewed. The CAD y flip leaves the page's visual
    direction unchanged, so this is also the CAD sweep direction.
    """
    radii = points - center
    cross = radii[:-1, 0] * radii[1:, 1] - radii[:-1, 1] * radii[1:, 0]
    return bool(cross.sum() > 0.0)


def _chord_length(points: NDArray[np.float64]) -> float:
    return float(np.hypot(*(points[-1] - points[0])))


def _chord_deviation(points: NDArray[np.float64]) -> float:
    """Largest distance from the chord, the run's total bow."""
    start, chord = points[0], points[-1] - points[0]
    if _chord_length(points) == 0.0:
        return float(np.hypot(*(points - start).T).max())
    t = np.clip(((points - start) @ chord) / (chord @ chord), 0.0, 1.0)
    return float(np.hypot(*(points - (start + t[:, None] * chord)).T).max())

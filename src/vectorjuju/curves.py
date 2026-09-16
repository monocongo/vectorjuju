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
actually runs; see ``CircleFit.clockwise`` for the endpoint mapping ezdxf's
counter-clockwise ARC needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.measure import CircleModel

from vectorjuju.tracing import _SIMPLIFY_PX, _scaled

# Circles need four points to say anything: any three non-collinear points
# determine one exactly, so their residual is always zero. Above that, this is
# the prior art's rule -- clean traced arcs land near 0.001 and the prior
# art's L-corner measured ~13, so 0.03 separates them by two orders of
# magnitude. It is the scale of the suite's 3 % radius tolerance, not an
# implication of it: a short, noisy arc's radius can be less certain than its
# residual suggests.
_MAX_RESIDUAL_RATIO = 0.03

Kind = Literal["line", "curve"]


@dataclass(frozen=True)
class CircleFit:
    """Least-squares circle through one run, in raster px (y down)."""

    center_px: tuple[float, float]
    radius_px: float
    clockwise: bool
    """Whether the run sweeps clockwise on the page; raster and CAD agree.

    Raster and CAD show the same page direction (``px_to_cad`` reflects y),
    but ezdxf ``add_arc`` always sweeps counter-clockwise in CAD coordinates:
    emit a clockwise run with its endpoint angles swapped -- ``start_angle``
    from the last run point, ``end_angle`` from the first.
    """


def classify(points_px: ArrayLike, dpi: float) -> tuple[Kind, CircleFit | None]:
    """Classify one traced run, in raster px, as a line or a curve.

    ``("line", None)`` for a straight run, ``("curve", fit)`` for a circular
    arc (emit ARC), and ``("curve", None)`` for a bent run that does not fit a
    circle (emit the SPLINE fallback). ``dpi`` scales the line/curve bow gate
    the same way ``trace_runs`` scales its simplification tolerance, so it has
    to be the dpi the run was traced at.

    A closed run (first point equals last) fits like any other; the writer has
    to emit it as a closed entity rather than an ARC between two coincident
    angles.

    Coordinates finite enough to pass validation can still overflow the
    chord and fit arithmetic's squared terms; that degrades to the SPLINE
    fallback rather than raising or trusting a garbage result, regardless of
    the caller's numpy floating-point error mode.
    """
    points = np.asarray(points_px, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"points_px must be an (N, 2) array with N >= 2, got shape {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_px must be finite")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"dpi must be a positive finite number, got {dpi!r}")

    try:
        # The tracer simplified this run with _SIMPLIFY_PX at the reference
        # DPI; a bend below that tolerance is indistinguishable from straight
        # linework.
        if _chord_deviation(points) <= _scaled(_SIMPLIFY_PX, dpi):
            return "line", None
        return "curve", _circle_fit(points)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return "curve", None


def _circle_fit(points: NDArray[np.float64]) -> CircleFit | None:
    """Fit a circle when the run really is one, else None.

    Three points determine a circle exactly, so a three-point run has no
    residual evidence and goes to the SPLINE fallback instead of a fabricated
    ARC centre. The radius sanity check keeps a fit that cannot span the run's
    chord (an arc's chord is never longer than its diameter) out. A run whose
    sweep direction is ambiguous or reverses also falls back rather than
    fabricate a fit.
    """
    if len(points) < 4:
        return None
    model = CircleModel.from_estimate(points)
    if not model:  # FailedEstimation: the points are collinear
        return None
    center = np.asarray(model.center, dtype=float)
    radius = float(model.radius)
    residual = float(np.abs(model.residuals(points)).max())
    if not (np.isfinite(radius) and np.isfinite(center).all() and np.isfinite(residual)):
        return None
    if residual > _MAX_RESIDUAL_RATIO * radius:
        return None
    if radius < 0.5 * _chord_length(points):
        return None
    clockwise = _sweep_direction(points, center)
    if clockwise is None:
        return None
    return CircleFit(
        center_px=(float(center[0]), float(center[1])),
        radius_px=radius,
        clockwise=clockwise,
    )


def _sweep_direction(points: NDArray[np.float64], center: NDArray[np.float64]) -> bool | None:
    """Clockwise/counter-clockwise sweep about centre, or None if not one directed arc.

    Unwraps each point's angle from centre and requires every step to turn
    the same way. A reversing run (out and back on the same circle) and a
    step wide enough to read as either the minor or the major arc both look
    non-monotonic here, so both refuse a fit instead of guessing a direction.
    """
    radii = points - center
    angles = np.unwrap(np.arctan2(radii[:, 1], radii[:, 0]))
    steps = np.sign(np.diff(angles))
    steps = steps[steps != 0]
    if len(steps) == 0 or not np.all(steps == steps[0]):
        return None
    return bool(steps[0] > 0)


def _chord_length(points: NDArray[np.float64]) -> float:
    return float(np.hypot(*(points[-1] - points[0])))


def _chord_deviation(points: NDArray[np.float64]) -> float:
    """Largest distance from the chord, the run's total bow."""
    start, chord = points[0], points[-1] - points[0]
    if _chord_length(points) == 0.0:
        return float(np.hypot(*(points - start).T).max())
    t = np.clip(((points - start) @ chord) / (chord @ chord), 0.0, 1.0)
    return float(np.hypot(*(points - (start + t[:, None] * chord)).T).max())

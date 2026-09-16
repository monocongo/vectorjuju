"""Calibrate real-world units per pixel from parsed calls against traced runs.

A bearing/distance call's printed distance and the traced run it bound to
(``vectorjuju.text.bind_calls``) measure the same physical edge twice: once in
survey units, once in raster pixels. Their ratio is one sample of
units-per-pixel; RANSAC fits the ratio the most samples agree with, rejecting
a call bound to the wrong run or one whose run does not span it. Fewer than
three usable samples, or a best consensus of one, raises
``ScaleCalibrationError`` -- there is no pixel-unit fallback.

A traced run stops short of its corners: the skeleton's junction vertex sits
inside the stroke, not at the centreline intersection, which is the slop the
acceptance suite's 8 px end-reach gates allow. Its raw length therefore
overshoots the scale by a few percent, so each end is first extended to the
intersection with the neighbouring bound run's centreline -- corner-to-corner
is what a recorded distance measures. That extension is measurement-only;
the emitted geometry keeps the tracer's own endpoints.

Nothing unbound reaches here, so curve-table cells (which never bind: a
radius has no bearing, and their borders are rejected by the tracer) cannot
pollute the fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from vectorjuju.convert import _require_fpp
from vectorjuju.errors import VectorjujuError
from vectorjuju.text import BoundCall
from vectorjuju.tracing import Run, _end_direction

_REFERENCE_DPI = 200.0
# Prior art: RANSAC needs at least three bound distance calls -- a consensus
# of one or two is not a regression (docs/prior-art/HANDOFF.md, "need >= 3").
_MIN_SAMPLES = 3
# Two agreeing calls are the smallest consensus that says anything.
_MIN_CONSENSUS = 2
# Inlier gate, relative to each call's own recorded distance: absorbs
# 2-decimal call rounding and residual corner slop while still rejecting a
# call bound to the wrong run, which is off by far more.
_INLIER_REL_TOL = 0.03
# Local-optimization passes after the consensus hypothesis: the inlier set can
# only grow, so this converges in a few passes; the cap makes termination
# independent of any floating-point tie-break.
_MAX_REFINEMENTS = 8
# Skeleton junction vertices stop ~5-8 px short of a corner at the reference
# dpi (A7's end-reach slop), and the neighbouring run is short too, so closing
# an end may travel up to the sum of both shortfalls. 16 px covers that with
# headroom while staying far inside the distance between real boundary runs.
_CORNER_EXTEND_PX = 16.0
# A neighbour whose centreline is nearly parallel is a merged dash or an
# offset line, not a corner; the tracer's own corner threshold is 15 degrees.
_CORNER_SIN = math.sin(math.radians(15.0))


class ScaleCalibrationError(VectorjujuError):
    """Scale cannot be recovered from the bound calls and no override was given."""


@dataclass(frozen=True)
class Scale:
    """Units per raster pixel and how they were obtained.

    ``value`` is the ratio of the parsed calls' own distance unit (feet on the
    sheets in scope) to traced pixels, or, with an override, exactly the value
    the caller supplied.
    """

    value: float
    method: Literal["ransac", "override"]


def _scaled(px_at_reference_dpi: float, dpi: float) -> float:
    return px_at_reference_dpi * dpi / _REFERENCE_DPI


def _closest_point(point: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance to a polyline, its nearest point, and that segment's unit direction."""
    best = (math.inf, point, np.zeros(2))
    for a, b in pairwise(points):
        edge = b - a
        length = float(np.hypot(*edge))
        if length == 0:
            continue
        t = max(0.0, min(1.0, float((point - a) @ edge) / (length * length)))
        nearest = a + t * edge
        distance = float(np.linalg.norm(point - nearest))
        if distance < best[0]:
            best = (distance, nearest, edge / length)
    return best


def _closes_end(points: np.ndarray, at_start: bool, neighbours: Sequence[np.ndarray], cap: float) -> np.ndarray:
    """Extend one run end to its corner.

    Intersects the run's outward end ray with each neighbour's centreline and
    takes the smallest valid crossing; the end is returned unchanged when no
    neighbour meets it inside the cap.
    """
    end = points[0] if at_start else points[-1]
    direction = _end_direction(points, at_start)
    if direction is None:
        return end
    best: tuple[float, np.ndarray | None] = (math.inf, None)
    for neighbour in neighbours:
        distance, nearest, edge = _closest_point(end, neighbour)
        if distance > 2 * cap or abs(direction[0] * edge[1] - direction[1] * edge[0]) < _CORNER_SIN:
            continue
        try:
            along, across = np.linalg.solve(
                np.array([[direction[0], -edge[0]], [direction[1], -edge[1]]]), nearest - end
            )
        except np.linalg.LinAlgError:  # parallel, already guarded; keep the degenerate safe
            continue
        if 0.0 < along <= cap and abs(across) <= cap and along < best[0]:
            best = (float(along), end + along * direction)
    return best[1] if best[1] is not None else end


def _closed_run_length_px(run: Run, runs: Sequence[Run], dpi: float) -> float:
    """Run length corner to corner: each end closed onto the neighbouring run.

    ponytail: closure is best-effort -- an end whose neighbour never bound
    (no call of its own) stays raw, and RANSAC then votes that sample out.
    """
    points = run.points_px
    cap = _scaled(_CORNER_EXTEND_PX, dpi)
    neighbours = [other.points_px for other in runs if other is not run]
    closed = points.copy()
    closed[0] = _closes_end(points, True, neighbours, cap)
    closed[-1] = _closes_end(points, False, neighbours, cap)
    return sum(math.dist(a, b) for a, b in pairwise(closed))


def _usable_samples(bound_calls: Sequence[BoundCall], dpi: float) -> list[tuple[float, float]]:
    """(run length px, call distance) for every call that can calibrate.

    A curve reference carries no distance, and a run with no measurable length
    divides by zero; both are skipped, not guessed at.
    """
    runs = [bound.run for bound in bound_calls]
    samples = []
    for bound in bound_calls:
        distance = bound.call.distance_ft
        if bound.call.kind != "bearing_distance" or distance is None:
            continue
        length = _closed_run_length_px(bound.run, runs, dpi)
        if math.isfinite(distance) and distance > 0 and math.isfinite(length) and length > 0:
            samples.append((length, distance))
    return samples


def _consensus(samples: Sequence[tuple[float, float]], fpp: float) -> list[tuple[float, float]]:
    return [sample for sample in samples if abs(sample[0] * fpp - sample[1]) <= _INLIER_REL_TOL * sample[1]]


def _fit(samples: Sequence[tuple[float, float]]) -> float:
    """Least-squares units-per-pixel through the origin (feet = k * px)."""
    return sum(length * distance for length, distance in samples) / sum(length * length for length, _ in samples)


def _ransac_fpp(samples: Sequence[tuple[float, float]]) -> tuple[float, int]:
    """Through-origin RANSAC over the samples; returns (fpp, consensus size).

    The model has one parameter (the ratio), so every sample is itself a
    minimal hypothesis: trying each in turn is exhaustive RANSAC -- no random
    subset can find a hypothesis this misses -- and deterministic, which the
    sidecar's byte-identity gate (A8) needs. Ties keep the earlier sample's
    hypothesis.
    """
    best_key, best_inliers = (0, 0.0), []
    for length, distance in samples:
        fpp = distance / length
        inliers = _consensus(samples, fpp)
        residual = sum((l * fpp - d) ** 2 for l, d in inliers)
        key = (len(inliers), -residual)
        if key > best_key:
            best_key, best_inliers = key, inliers

    for _ in range(_MAX_REFINEMENTS):
        fpp = _fit(best_inliers)
        grown = _consensus(samples, fpp)
        if len(grown) == len(best_inliers):
            break
        best_inliers = grown
    return fpp, len(best_inliers)


def calibrate_scale(bound_calls: Sequence[BoundCall], *, scale: float | None = None, dpi: float = 200.0) -> Scale:
    """Recover units per raster pixel from ``bound_calls``, or accept ``scale``.

    ``scale`` skips calibration outright -- even when the calls on hand cannot
    calibrate -- and is recorded as ``method="override"``. Otherwise each
    usable call votes with its recorded distance over its corner-to-corner run
    length; the winning consensus is refit least-squares and returned with
    ``method="ransac"``. Raises ``ScaleCalibrationError`` when the calls are
    too few or disagree: never a pixel-unit fallback.
    """
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"calibrate_scale requires a finite dpi > 0, got {dpi!r}")
    if scale is not None:
        return Scale(value=_require_fpp(scale, "calibrate_scale"), method="override")

    samples = _usable_samples(bound_calls, dpi)
    if len(samples) < _MIN_SAMPLES:
        raise ScaleCalibrationError(
            f"only {len(samples)} usable bound distance call(s): need >= {_MIN_SAMPLES} to calibrate "
            "scale, and no scale override was given"
        )

    fpp, consensus = _ransac_fpp(samples)
    if consensus < _MIN_CONSENSUS:
        raise ScaleCalibrationError(
            f"no {_MIN_CONSENSUS} of {len(samples)} usable calls agree on a scale (best consensus: {consensus})"
        )
    return Scale(value=fpp, method="ransac")

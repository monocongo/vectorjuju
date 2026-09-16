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
is what a recorded distance measures. ``tracing.join_corners`` applies the
same closure to the geometry ``convert()`` emits, so the drawing and this
measurement agree. A cell index keeps each end's search on the runs whose
geometry is actually near it, so closure costs per run, not per run pair.

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
from vectorjuju.tracing import _CORNER_CLOSE_PX, Run, _cell_index, _closes_end, _nearby_runs

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


def _closed_run_length_px(
    run_index: int, runs: Sequence[Run], index: dict[tuple[int, int], list[int]], cap: float
) -> float:
    """Run length corner to corner: each end closed onto the neighbouring run.

    ponytail: closure is best-effort -- an end whose neighbour never bound
    (no call of its own) stays raw, and RANSAC then votes that sample out.
    """
    points = runs[run_index].points_px
    if len(points) < 2:
        return 0.0  # nothing measurable to close; the caller's length > 0 gate skips it
    cell = 2 * cap
    closed = points.copy()
    closed[0] = _closes_end(points, True, _nearby_runs(index, cell, points[0], runs, run_index), cap)
    closed[-1] = _closes_end(points, False, _nearby_runs(index, cell, points[-1], runs, run_index), cap)
    return sum(math.dist(a, b) for a, b in pairwise(closed))


def _usable_samples(bound_calls: Sequence[BoundCall], dpi: float) -> list[tuple[float, float]]:
    """(run length px, call distance) for every call that can calibrate.

    A curve reference carries no distance, and a run with no measurable length
    divides by zero; both are skipped, not guessed at.
    """
    runs = [bound.run for bound in bound_calls]
    cap = _scaled(_CORNER_CLOSE_PX, dpi)
    index = _cell_index(runs, 2 * cap)
    samples = []
    for i, bound in enumerate(bound_calls):
        distance = bound.call.distance_ft
        if bound.call.kind != "bearing_distance" or distance is None:
            continue
        length = _closed_run_length_px(i, runs, index, cap)
        if math.isfinite(distance) and distance > 0 and math.isfinite(length) and length > 0:
            samples.append((length, distance))
    return samples


def _consensus(lengths: np.ndarray, distances: np.ndarray, fpp: float) -> np.ndarray:
    """Boolean mask of the samples whose own recorded distance accepts ``fpp``."""
    return np.abs(lengths * fpp - distances) <= _INLIER_REL_TOL * distances


def _fit(lengths: np.ndarray, distances: np.ndarray, inliers: np.ndarray) -> float:
    """Least-squares units-per-pixel through the origin (feet = k * px)."""
    return float(np.dot(lengths[inliers], distances[inliers]) / np.dot(lengths[inliers], lengths[inliers]))


def _ransac_fpp(samples: Sequence[tuple[float, float]]) -> tuple[float, np.ndarray]:
    """Through-origin RANSAC over the samples; returns (fpp, inlier mask).

    The model has one parameter (the ratio), so every sample is itself a
    minimal hypothesis: trying each in turn is exhaustive RANSAC -- no random
    subset can find a hypothesis this misses -- and deterministic, which the
    sidecar's byte-identity gate (A8) needs. Ties keep the earlier sample's
    hypothesis. Consensus and residual arithmetic is vectorized, so the
    quadratic term is the usable-call count (bounded by the text side's
    ``_MAX_RUNS_FOR_TEXT``) rather than every run's geometry.

    Refinement converges on the inlier membership itself, not its size: a
    swapped set of the same size is not a fixpoint, and the returned fit must
    be one its own reported inliers accept.
    """
    lengths = np.array([length for length, _ in samples])
    distances = np.array([distance for _, distance in samples])
    empty = np.zeros(len(samples), dtype=bool)

    best_key, best_inliers = (0, 0.0), empty
    for length, distance in zip(lengths.tolist(), distances.tolist(), strict=True):
        fpp = distance / length
        if not math.isfinite(fpp):
            continue  # finite operands, overflowing quotient: no sample can agree with it
        inliers = _consensus(lengths, distances, fpp)
        residual = float(np.sum((lengths[inliers] * fpp - distances[inliers]) ** 2))
        key = (int(inliers.sum()), -residual)
        if key > best_key:
            best_key, best_inliers = key, inliers

    if not best_inliers.any():
        return 0.0, empty  # every hypothesis was a non-finite ratio

    for _ in range(_MAX_REFINEMENTS):
        fpp = _fit(lengths, distances, best_inliers)
        if not math.isfinite(fpp):
            return 0.0, empty  # least-squares overflowed; no consensus to report
        grown = _consensus(lengths, distances, fpp)
        if np.array_equal(grown, best_inliers):
            return fpp, best_inliers
        best_inliers = grown
    fpp = _fit(lengths, distances, best_inliers)
    return (fpp, best_inliers) if math.isfinite(fpp) else (0.0, empty)


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

    fpp, inliers = _ransac_fpp(samples)
    consensus = int(inliers.sum())
    if consensus < _MIN_CONSENSUS or not math.isfinite(fpp):
        raise ScaleCalibrationError(
            f"no {_MIN_CONSENSUS} of {len(samples)} usable calls agree on a scale (best consensus: {consensus})"
        )
    return Scale(value=fpp, method="ransac")

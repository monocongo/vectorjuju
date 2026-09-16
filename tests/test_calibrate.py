"""U2 plus the calibration mechanism, no OCR: planted call distances against
planted pixel lengths recover the planted fpp; an override is exact; too few,
unmeasurable, or disagreeing calls raise instead of emitting a pixel-unit
guess.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vectorjuju import tracing
from vectorjuju.calibrate import Scale, ScaleCalibrationError, _consensus, _ransac_fpp, calibrate_scale
from vectorjuju.convert import VectorjujuError
from vectorjuju.synthetic_plat import (
    CURVE_EDGES,
    PAGE_H,
    PARCEL_FT,
    RENDER_DPI,
    SCALE_PT_PER_FT,
    bearing_distance,
    ft_to_pt,
)
from vectorjuju.text import BoundCall, TextItem, parse_call
from vectorjuju.tracing import Run

PLANTED_FPP = 72.0 / (SCALE_PT_PER_FT * RENDER_DPI)  # 0.423529 ft/px at 200 dpi


def _to_px(pt: tuple[float, float]) -> tuple[float, float]:
    scale = RENDER_DPI / 72.0
    return pt[0] * scale, (PAGE_H - pt[1]) * scale


def _bound_call(raw: str, points: list[tuple[float, float]]) -> BoundCall:
    call = parse_call(raw)
    assert call is not None  # the test's own raw text must parse
    run = Run(points_px=np.array(points, dtype=float))
    return BoundCall(
        call=call, run=run, item=TextItem(text=raw, box_px=(0.0, 0.0, 1.0, 1.0), source="crop"), distance_px=0.0
    )


def _call(length_px: float, distance_ft: float) -> BoundCall:
    """A horizontal run of ``length_px`` whose call records ``distance_ft``."""
    return _bound_call(f"N 0°00'00\" E  {distance_ft:.2f}'", [(0.0, 0.0), (length_px, 0.0)])


def _planted_straight_calls() -> list[BoundCall]:
    """The fixture's four straight edges, as the generator prints them and as
    they trace at ``RENDER_DPI``: distance in feet, run in pixels."""
    calls = []
    for i in range(len(PARCEL_FT)):
        if i in CURVE_EDGES:
            continue
        p0, p1 = PARCEL_FT[i], PARCEL_FT[(i + 1) % len(PARCEL_FT)]
        bearing, dist_ft = bearing_distance(p0, p1)
        calls.append(_bound_call(f"{bearing}  {dist_ft:.2f}'", [_to_px(ft_to_pt(p0)), _to_px(ft_to_pt(p1))]))
    return calls


def test_planted_call_distances_recover_the_planted_fpp() -> None:
    calls = _planted_straight_calls()
    assert len(calls) == 4  # the fixture plants four straight edges

    scale = calibrate_scale(calls)

    assert scale.method == "ransac"
    assert abs(scale.value - PLANTED_FPP) <= 0.01 * PLANTED_FPP


def test_noisy_inlier_runs_still_land_inside_the_scale_budget() -> None:
    # Each run is up to 1 % off the length its call implies -- endpoint and
    # simplification slop well inside the 3 % consensus gate -- so the fit
    # must average the noise, not adopt one sample's ratio.
    offsets = [1.01, 0.99, 1.005, 0.995]
    calls = [_call(200.0 + 40.0 * k, (200.0 + 40.0 * k) * PLANTED_FPP * o) for k, o in enumerate(offsets)]

    scale = calibrate_scale(calls)

    assert scale.method == "ransac"
    assert abs(scale.value - PLANTED_FPP) <= 0.01 * PLANTED_FPP


def test_ransac_rejects_a_call_bound_to_the_wrong_run() -> None:
    good = [_call(200.0 + 50.0 * k, (200.0 + 50.0 * k) * PLANTED_FPP) for k in range(4)]
    misbound = _call(100.0, 500.0)  # ratio 5.0 ft/px, ten times the planted scale

    scale = calibrate_scale([*good, misbound])

    assert scale.method == "ransac"
    assert math.isclose(scale.value, PLANTED_FPP, rel_tol=1e-3)


def test_run_length_follows_the_polyline_not_the_endpoint_chord() -> None:
    # A 3-4-5 path: 70 px of traced polyline, 50 px from endpoint to endpoint.
    # A chord-length implementation would return ~0.59, not the planted 0.42.
    legs = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0)]
    calls = [_bound_call(f"N 0°00'00\" E  {70.0 * PLANTED_FPP:.2f}'", legs) for _ in range(3)]

    scale = calibrate_scale(calls)

    assert abs(scale.value - PLANTED_FPP) <= 0.01 * PLANTED_FPP


def _trimmed_square_calls(offset: float = 0.0) -> list[BoundCall]:
    """A 400/300 px square trimmed 7 px per end, planted at 0.25 ft/px.

    The traced runs stop short of their corners (skeleton junction geometry,
    the slop A7's 8 px end-reach gate allows): on this square, raw ratios
    read +3.6 % to +4.9 % high. ``offset`` shifts the square so ends land on
    the closure cell index's cell edges as well as inside them.
    """
    corners = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    trim = 7.0
    calls = []
    for i in range(len(corners)):
        a = np.array(corners[i]) + offset
        b = np.array(corners[(i + 1) % len(corners)]) + offset
        direction = (b - a) / np.linalg.norm(b - a)
        full_ft = np.linalg.norm(b - a) * 0.25  # planted 0.25 ft/px
        calls.append(
            _bound_call(f"N 0°00'00\" E  {full_ft:.2f}'", [tuple(a + trim * direction), tuple(b - trim * direction)])
        )
    return calls


# Closing each end onto the neighbouring run's centreline has to recover the
# planted 0.25; the raw trimmed runs would read 3.6 % to 4.9 % high. 16 px
# puts the ends on the closure cell index's cell edges as well as inside them.
@pytest.mark.parametrize("offset", [0.0, 16.0])
def test_trimmed_runs_are_closed_corner_to_corner_before_measuring(offset: float) -> None:
    calls = _trimmed_square_calls(offset)

    scale = calibrate_scale(calls)

    assert abs(scale.value - 0.25) <= 0.01 * 0.25


def test_corner_closure_does_not_scan_distant_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # A distant run is past the closure cap and can never be a neighbour, so
    # the per-end scans must never touch it: at the text side's 4,000-run cap,
    # every end scanning every run's full polyline is the quadratic blow-up.
    far = _bound_call(f"N 0°00'00\" E  {100.0:.2f}'", [(6000.0, 6000.0), (6400.0, 6000.0)])
    scanned: list[np.ndarray] = []
    closest_point = tracing._closest_point

    def spy(point: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        scanned.append(points)
        return closest_point(point, points)

    monkeypatch.setattr(tracing, "_closest_point", spy)

    scale = calibrate_scale([*_trimmed_square_calls(), far])

    assert abs(scale.value - 0.25) <= 0.01 * 0.25
    assert not any(points is far.run.points_px for points in scanned)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_usable_calls_raise_instead_of_guessing(count: int) -> None:
    calls = [_call(100.0, 100.0 * PLANTED_FPP) for _ in range(count)]
    with pytest.raises(ScaleCalibrationError):
        calibrate_scale(calls)


def test_curve_refs_and_unmeasurable_runs_do_not_count_as_usable() -> None:
    good = [_call(100.0, 100.0 * PLANTED_FPP) for _ in range(3)]
    curve_ref = _bound_call("C1", [(0.0, 0.0), (100.0, 0.0)])
    zero_run = _call(0.0, 100.0)  # a call whose run has no measurable length
    empty_run = _bound_call(f"N 0°00'00\" E  {100.0:.2f}'", [])  # ...and one with no geometry at all

    # None of the fakes can calibrate, but the three real ones still do.
    scale = calibrate_scale([curve_ref, zero_run, empty_run, *good])
    assert abs(scale.value - PLANTED_FPP) <= 0.01 * PLANTED_FPP

    # Drop to two real samples and the unusable trio is not mistaken for a third.
    with pytest.raises(ScaleCalibrationError):
        calibrate_scale([curve_ref, zero_run, empty_run, *good[:2]])


def test_ratio_overflow_raises_instead_of_fitting_an_empty_consensus() -> None:
    # Both operands are finite and positive, but their quotient is not: no
    # hypothesis has a consensus, so this must fail as a calibration error
    # rather than divide by an empty inlier set.
    calls = [_call(1e-100, 1e250) for _ in range(3)]

    with pytest.raises(ScaleCalibrationError):
        calibrate_scale(calls)


def test_refinement_converges_on_membership_not_just_its_size() -> None:
    # The least-squares refit of one hypothesis's inliers moves the 3 % window
    # onto a different set of the same size; a same-size swap is not a
    # fixpoint, so the returned fit must be the one its reported inliers
    # accept -- not the superseded set's fit.
    samples = [(1.0, 4.06), (1.0, 1.09), (1.0, 1.07), (4.0, 1.11), (4.0, 4.56), (4.0, 4.46)]

    fpp, inliers = _ransac_fpp(samples)

    lengths = np.array([length for length, _ in samples])
    distances = np.array([distance for _, distance in samples])
    assert int(inliers.sum()) == 2
    assert np.array_equal(_consensus(lengths, distances, fpp), inliers)


def test_disagreeing_calls_raise_instead_of_averaging() -> None:
    calls = [_call(100.0, 100.0), _call(100.0, 500.0), _call(100.0, 2000.0)]  # ratios 1, 5, 20

    with pytest.raises(ScaleCalibrationError):
        calibrate_scale(calls)


def test_override_skips_calibration_and_is_recorded_exactly() -> None:
    scale = calibrate_scale([], scale=0.25)

    assert scale == Scale(value=0.25, method="override")


@pytest.mark.parametrize("bad", [0.0, -0.42, float("nan"), float("inf")])
def test_override_must_be_a_representable_scale(bad: float) -> None:
    with pytest.raises(ValueError):
        calibrate_scale([], scale=bad)


@pytest.mark.parametrize("dpi", [0.0, -1.0, float("nan"), float("inf")])
def test_calibration_requires_a_finite_dpi(dpi: float) -> None:
    with pytest.raises(ValueError):
        calibrate_scale([], dpi=dpi)


def test_calibration_is_deterministic() -> None:
    calls = _planted_straight_calls()
    assert calibrate_scale(calls) == calibrate_scale(calls)


def test_scale_calibration_error_is_a_vectorjuju_error() -> None:
    assert issubclass(ScaleCalibrationError, VectorjujuError)

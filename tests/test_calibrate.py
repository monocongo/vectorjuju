"""U2 plus the calibration mechanism, no OCR: planted call distances against
planted pixel lengths recover the planted fpp; an override is exact; too few,
unmeasurable, or disagreeing calls raise instead of emitting a pixel-unit
guess.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vectorjuju.calibrate import Scale, ScaleCalibrationError, calibrate_scale
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


def test_trimmed_runs_are_closed_corner_to_corner_before_measuring() -> None:
    # A traced run stops short of its corners (skeleton junction geometry, the
    # slop A7's 8 px end-reach gate allows). On a 400/300 px square trimmed
    # 7 px per end, raw ratios read +3.6 % to +4.9 % high; closing each end
    # onto the neighbouring run's centreline has to recover the planted 0.25.
    corners = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    trim = 7.0
    calls = []
    for i in range(len(corners)):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % len(corners)])
        edge = b - a
        direction = edge / np.linalg.norm(edge)
        full_ft = np.linalg.norm(edge) * 0.25  # planted 0.25 ft/px
        calls.append(
            _bound_call(f"N 0°00'00\" E  {full_ft:.2f}'", [tuple(a + trim * direction), tuple(b - trim * direction)])
        )

    scale = calibrate_scale(calls)

    assert abs(scale.value - 0.25) <= 0.01 * 0.25


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_usable_calls_raise_instead_of_guessing(count: int) -> None:
    calls = [_call(100.0, 100.0 * PLANTED_FPP) for _ in range(count)]
    with pytest.raises(ScaleCalibrationError):
        calibrate_scale(calls)


def test_curve_refs_and_unmeasurable_runs_do_not_count_as_usable() -> None:
    good = [_call(100.0, 100.0 * PLANTED_FPP) for _ in range(3)]
    curve_ref = _bound_call("C1", [(0.0, 0.0), (100.0, 0.0)])
    zero_run = _call(0.0, 100.0)  # a call whose run has no measurable length

    # Neither fake sample can calibrate, but the three real ones still do.
    scale = calibrate_scale([curve_ref, zero_run, *good])
    assert abs(scale.value - PLANTED_FPP) <= 0.01 * PLANTED_FPP

    # Drop to two real samples and the unusable pair is not mistaken for a third.
    with pytest.raises(ScaleCalibrationError):
        calibrate_scale([curve_ref, zero_run, *good[:2]])


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

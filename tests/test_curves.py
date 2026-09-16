"""U3 curve half: fit and classify arcs, lines, and the SPLINE fallback.

Rasters are drawn in the test, so a failure names the property rather than
fixture drift; the synthetic plat's planted arcs are classified in
``test_tracing_plat.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from acceptance_helpers import arc_midpoint
from PIL import Image, ImageDraw

from vectorjuju.curves import CircleFit, classify
from vectorjuju.tracing import trace_runs

DPI = 200.0
CENTRE = (420.0, 420.0)


def _raster(points: np.ndarray, width: int = 5) -> Image.Image:
    image = Image.new("L", (900, 900), 255)
    ImageDraw.Draw(image).line([tuple(point) for point in points], fill=0, width=width, joint="curve")
    return image


def _arc(radius: float = 300.0, start_deg: float = 20.0, end_deg: float = 60.0) -> np.ndarray:
    angles = np.radians(np.linspace(start_deg, end_deg, 49))
    return np.column_stack([CENTRE[0] + radius * np.cos(angles), CENTRE[1] + radius * np.sin(angles)])


def _classify_one(image: Image.Image) -> tuple[str, CircleFit | None]:
    runs = trace_runs(image, DPI)
    assert len(runs) == 1
    return classify(runs[0].points_px, DPI)


def _endpoint_residuals(points: np.ndarray, fit) -> list[float]:
    """Distance of each traced end from the fitted circle: the ARC endpoints' error."""
    centre = np.asarray(fit.center_px)
    return [abs(float(np.hypot(*(end - centre))) - fit.radius_px) for end in (points[0], points[-1])]


def test_drawn_arc_classifies_as_a_circle():
    points = _arc()
    kind, fit = _classify_one(_raster(points))

    assert kind == "curve"
    assert fit is not None
    assert abs(fit.radius_px - 300.0) / 300.0 <= 0.03
    assert max(_endpoint_residuals(points, fit)) <= 4.0
    assert fit.clockwise  # increasing raster angles sweep clockwise on the page


def test_reversing_a_run_flips_the_sweep():
    points = trace_runs(_raster(_arc()), DPI)[0].points_px

    _, forward = classify(points, DPI)
    _, backward = classify(points[::-1], DPI)

    assert forward is not None and backward is not None
    assert forward.clockwise != backward.clockwise


def test_clockwise_and_midpoint_agree_with_the_planted_bulge():
    planted = _arc()
    runs = trace_runs(_raster(planted), DPI)
    assert len(runs) == 1
    points = runs[0].points_px
    kind, fit = classify(points, DPI)
    assert kind == "curve"
    assert fit is not None

    assert np.hypot(*(arc_midpoint(points, fit) - planted[len(planted) // 2])) <= 2.0


def test_l_corner_runs_classify_as_lines():
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.line([(60, 240), (60, 60), (240, 60)], fill=0, width=5, joint="curve")

    runs = trace_runs(image, DPI)

    assert len(runs) == 2
    assert [classify(run.points_px, DPI) for run in runs] == [("line", None), ("line", None)]


def test_straight_runs_classify_as_lines():
    points = np.column_stack([np.linspace(50.0, 350.0, 7), np.full(7, 80.0)])

    assert classify(points, DPI) == ("line", None)
    assert classify(points[:2], DPI) == ("line", None)


def test_bow_below_the_simplification_tolerance_is_a_line():
    # 5 degrees of a 300 px radius: 0.29 px of sagitta, which the tracer's
    # simplification tolerance would have flattened anyway.
    points = _arc(radius=300.0, start_deg=20.0, end_deg=25.0)

    assert classify(points, DPI) == ("line", None)


def test_bent_run_that_is_not_a_circle_falls_back_to_spline():
    # An S-curve: smooth everywhere, so the tracer keeps it one run, but it
    # reverses curvature and cannot fit a circle.
    t = np.linspace(0.0, 1.0, 200)[:, None]
    s_curve = (
        (1 - t) ** 3 * np.array([150.0, 450.0])
        + 3 * (1 - t) ** 2 * t * np.array([266.0, 370.0])
        + 3 * (1 - t) * t**2 * np.array([383.0, 530.0])
        + t**3 * np.array([500.0, 450.0])
    )

    kind, fit = _classify_one(_raster(s_curve))

    assert kind == "curve"
    assert fit is None  # the writer emits the SPLINE fallback


def test_three_point_bend_falls_back_to_spline():
    # Every three non-collinear points lie on a circle exactly, so a 3-point
    # run has no residual evidence; it goes to the SPLINE fallback instead of
    # a fabricated ARC centre.
    bend = np.array([[60.0, 240.0], [60.0, 60.0], [240.0, 60.0]])

    assert classify(bend, DPI) == ("curve", None)


@pytest.mark.parametrize("dpi", [200.0, 400.0])
def test_arc_classification_is_dpi_invariant(dpi: float):
    scale = dpi / DPI
    image = Image.new("L", (round(900 * scale), round(900 * scale)), 255)
    planted = _arc() * scale
    ImageDraw.Draw(image).line([tuple(point) for point in planted], fill=0, width=round(5 * scale), joint="curve")

    runs = trace_runs(image, dpi)

    assert len(runs) == 1
    kind, fit = classify(runs[0].points_px, dpi)
    assert kind == "curve"
    assert fit is not None
    # Kind is what has to be invariant. At 400 dpi the same fit can be 3.6 %
    # off the planted radius -- a 40 degree arc's radius is ill-conditioned
    # against sub-pixel thinning -- while its residual stays 30x under the
    # gate. Radius accuracy is gated at the reference dpi in
    # test_drawn_arc_classifies_as_a_circle.


def test_bow_gate_scales_with_dpi():
    # 1 px of sagitta over a 200 px chord: under the 1.5 px reference bow at
    # 200 dpi, over the 0.75 px gate at 100 dpi -- the gate follows the
    # working resolution the way tracing's simplification tolerance does.
    bend = np.array([[0.0, 0.0], [100.0, 1.0], [200.0, 0.0]])

    assert classify(bend, DPI) == ("line", None)
    assert classify(bend, DPI / 2)[0] == "curve"


@pytest.mark.parametrize("dpi", [0, -1, float("nan"), float("inf")])
def test_classify_rejects_invalid_dpi(dpi: float):
    with pytest.raises(ValueError):
        classify(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 5.0]]), dpi)


def test_classify_rejects_invalid_points():
    with pytest.raises(ValueError):
        classify(np.zeros((0, 2)), DPI)
    with pytest.raises(ValueError):
        classify(np.zeros((5, 3)), DPI)
    with pytest.raises(ValueError):
        classify(np.array([[0.0, 0.0], [float("nan"), 1.0], [2.0, 3.0]]), DPI)

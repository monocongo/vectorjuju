"""Unit gates for tracing: the straight half of U3, on rasters drawn in the test.

Nothing here needs OCR or the sheet fixture; each test draws the one property it
pins, so a failure names the property rather than a fixture drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from vectorjuju.tracing import _merge_dashes, trace_runs

DPI = 200.0


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (width, height), 255)
    return image, ImageDraw.Draw(image)


@pytest.mark.parametrize("width", [3, 4])
def test_thick_stroke_traces_one_centreline(width: int):
    image, draw = _canvas(120, 60)
    draw.line((10, 30, 110, 30), fill=0, width=width)

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    points = runs[0].points_px
    assert np.all(np.abs(points[:, 1] - 30.0) <= 1.5)  # one centreline, not two edges
    assert np.ptp(points[:, 0]) >= 95


def test_l_corner_splits_into_two_straight_runs():
    image, draw = _canvas(120, 120)
    draw.line((30, 20, 30, 90), fill=0, width=5)
    draw.line((30, 90, 100, 90), fill=0, width=5)

    runs = trace_runs(image, DPI)

    assert len(runs) == 2
    corner = np.array([30.0, 90.0])
    for run in runs:
        points = run.points_px
        assert np.ptp(points[:, 0]) < 3 or np.ptp(points[:, 1]) < 3
        assert min(np.hypot(*(points[0] - corner)), np.hypot(*(points[-1] - corner))) <= 4.0
        assert np.sum(np.hypot(*np.diff(points, axis=0).T)) >= 60.0  # not truncated at the corner


def test_single_stroke_l_corner_splits():
    # One polyline, so simplification keeps a single corner vertex: the split
    # must not depend on the corner arriving as two quantised vertices.
    image, draw = _canvas(300, 300)
    draw.line([(60, 240), (60, 60), (240, 60)], fill=0, width=5, joint="curve")

    assert len(trace_runs(image, DPI)) == 2


def test_shallow_deflection_splits():
    # A 20 degree deflection between long straight runs is a corner, not the
    # curvature of an arc; the adjacent-vertex test must keep splitting it.
    image, draw = _canvas(600, 300)
    draw.line([(50, 150), (300, 150), (550, 150 - 250 * np.tan(np.radians(20)))], fill=0, width=5, joint="curve")

    assert len(trace_runs(image, DPI)) == 2


def test_solid_and_empty_rasters_return_no_runs():
    assert trace_runs(Image.new("L", (100, 100), 0), DPI) == []
    assert trace_runs(Image.new("L", (100, 100), 255), DPI) == []
    assert trace_runs(np.zeros((0, 0), np.uint8), DPI) == []


def test_short_thick_run_survives():
    image, draw = _canvas(80, 80)
    draw.line((40, 25, 40, 45), fill=0, width=5)  # 20 px long, 5 px wide

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    assert np.ptp(runs[0].points_px[:, 1]) >= 14


def test_glyphs_yield_no_primitives():
    image, draw = _canvas(160, 60)
    draw.text((5, 5), "ABC 12", font=ImageFont.load_default(size=20), fill=0)

    assert trace_runs(image, DPI) == []


def test_dashed_line_merges_into_one_run():
    image, draw = _canvas(240, 40)
    for k in range(10):
        x = 10 + k * 20
        draw.line((x, 20, x + 10, 20), fill=0, width=4)

    runs = trace_runs(image, DPI)

    assert len(runs) == 1
    assert np.ptp(runs[0].points_px[:, 0]) >= 170  # spans the gaps


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
def test_one_pixel_edge_line_is_not_a_boundary(edge: str):
    """A 1 px line on a raster edge must not measure wide.

    Cross-section samples outside the raster used to clip onto the edge pixel,
    counting that one row once per sample and lifting the line over the
    boundary-width gate.
    """
    image, draw = _canvas(240, 80)
    if edge in ("top", "bottom"):
        y = 0 if edge == "top" else 79
        draw.line((10, y, 230, y), fill=0, width=1)
    else:
        x = 0 if edge == "left" else 239
        draw.line((x, 10, x, 70), fill=0, width=1)

    assert trace_runs(image, DPI) == []


def test_a_crowded_cell_does_not_starve_a_distant_dash_pair():
    """Forty fragments in one grid cell must not hide a chain 200 px away.

    Endpoint candidates are capped to the nearest few per endpoint so a
    crowded cell cannot enumerate every pair inside it; the cap must keep the
    nearest candidates, which are the gaps greedy merging wants first.
    """
    coverage = np.zeros((400, 400), np.float32)
    centre = np.array([196.0, 196.0])
    crowd = [
        (np.array([centre, centre + 20.0 * np.array([np.cos(angle), np.sin(angle)])]), 4.0)
        for angle in np.radians(np.arange(40) * 9.0)
    ]
    entries = crowd + [
        (np.array([[0.0, 0.0], [40.0, 0.0]]), 4.0),
        (np.array([[46.0, 0.0], [86.0, 0.0]]), 4.0),
    ]

    merged = _merge_dashes(entries, coverage, DPI)

    assert len(merged) == len(_merge_dashes(crowd, coverage, DPI)) + 1  # the pair merged


def test_run_count_is_dpi_invariant():
    def square(dpi: float) -> Image.Image:
        scale = dpi / DPI
        image, draw = _canvas(round(200 * scale), round(200 * scale))
        margin = round(40 * scale)
        draw.rectangle(
            (margin, margin, image.width - margin, image.height - margin),
            outline=0,
            width=max(2, round(4 * scale)),
        )
        return image

    for dpi in (DPI / 2, DPI, 2 * DPI):
        assert len(trace_runs(square(dpi), dpi)) == 4


def test_parallel_dashed_lines_do_not_merge():
    image, draw = _canvas(400, 60)
    x = 10
    while x + 30 < 390:
        draw.line((x, 20, x + 30, 20), fill=0, width=4)
        draw.line((x, 28, x + 30, 28), fill=0, width=4)
        x += 40

    runs = trace_runs(image, DPI)

    assert len(runs) == 2  # each line merges along its dashes, not across the 8 px gap
    assert sorted(round(float(np.median(run.points_px[:, 1]))) for run in runs) == [20, 28]


def test_stroke_width_scales_with_dpi():
    for dpi in (200.0, 600.0):
        image, draw = _canvas(round(400 * dpi / DPI), round(100 * dpi / DPI))
        draw.line((30, 50 * dpi / DPI, 370 * dpi / DPI, 50 * dpi / DPI), fill=0, width=round(1.4 * dpi / 72))

        assert len(trace_runs(image, dpi)) == 1


def test_exclude_mask_removes_masked_stroke():
    image, draw = _canvas(240, 60)
    draw.line((10, 15, 230, 15), fill=0, width=5)
    draw.line((10, 45, 230, 45), fill=0, width=5)
    mask = np.zeros((60, 240), bool)
    mask[35:56, :] = True

    runs = trace_runs(image, DPI, exclude_mask=mask)

    assert len(runs) == 1
    assert np.median(runs[0].points_px[:, 1]) < 30  # the surviving line, not the masked one


def test_excluded_ink_does_not_widen_a_surviving_stroke():
    image, draw = _canvas(300, 60)
    draw.line((10, 30, 290, 30), fill=0, width=2)  # thinner than the gate on its own
    draw.rectangle((30, 22, 280, 29), fill=0)  # masked text ink touching the line
    mask = np.zeros((60, 300), bool)
    mask[22:30, 30:281] = True

    assert trace_runs(image, DPI, exclude_mask=mask) == []


def test_invalid_inputs_are_rejected():
    image, draw = _canvas(40, 40)
    draw.line((5, 20, 35, 20), fill=0, width=4)

    for dpi in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            trace_runs(image, dpi=dpi)
    with pytest.raises(ValueError):
        trace_runs(np.zeros((40, 40, 3), np.uint8), DPI)
    with pytest.raises(ValueError):
        trace_runs(np.zeros((40, 40), np.float64), DPI)
    with pytest.raises(ValueError):
        trace_runs(image, DPI, exclude_mask=np.zeros((1, 1), bool))
